"""Runtime evidence cannot be supplied by an unrelated endpoint or stale launch."""
import asyncio
from copy import deepcopy
import hashlib
import json

import httpx
import pytest

from test_local_demo_supervisor import rig
from test_local_moulsot_bridge import bridge, result as upstream_result
from engine.config import ROOT, load_config
from engine.stt import SpeechToText, STTError
from engine.asr_provenance import decode_snapshot, runtime_ack, validate_snapshot


ENV = 'MOULSOT_RUNTIME_SNAPSHOT'
LOCAL = 'http://127.0.0.1:8012/transcribe'
SECRET = 'PRIVATE_PATH_TOKEN_SENTINEL'


@pytest.fixture
def snapshot():
    return dict(schema_version=1, run_id='a' * 32, verified_at='2026-09-06T10:00:00Z',
        source_model='atlasia/moulsot.v0.3', model_variant='moulsot.v0.3-Q4_K_M-mmproj-Q8_0',
        conversion_revision='ca66fea7f3db516212720fd005a0d70a57213de8', runtime_release='b10809',
        selected_device='cpu', verification='launch_sha256_and_extracted_runtime',
        decoder_sha256='9534be065f1990a0f8c4159ee16531cbefc5b0a1f9dcec47c57ea3088c3f9e78',
        projector_sha256='d530672efeaee3a0dad721334019e9155d2cea6184eeb126834fa6df2a8aa26c',
        runtime_archive_sha256='c77bfcd9ed8d91e8721a2d6a290b907fddd4fa5412a47b21c6fa1709116b85f9',
        cuda_archive_sha256='8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6')


@pytest.fixture
def config(monkeypatch, snapshot):
    monkeypatch.setenv(ENV, json.dumps(snapshot))
    monkeypatch.setenv('MOULSOT_PROTOCOL', 'json')
    return load_config(ROOT / 'configs/pizza.json')


def adapter(config, tmp_path, client, **changes):
    options = dict(primary='moulsot', fallback=False, endpoint=LOCAL,
        client=client, api_key='offline-fixture', hf_token='offline-hf')
    options.update(changes)
    return SpeechToText(config, tmp_path, **options)


def reply(snapshot, **changes):
    payload = {'text': 'unchanged raw transcript', 'confidence': None, 'runtime_snapshot': runtime_ack(snapshot),
        'provenance': {'upstream_model': 'C:/' + SECRET + '/model.gguf', 'verified': True, SECRET: SECRET}}
    payload.update(changes)
    return httpx.Response(200, json=payload)


def safe_calls(stt):
    assert SECRET not in json.dumps(stt.calls)
    assert all(call['runtime_provenance']['actual_device'] is None for call in stt.calls)


@pytest.mark.parametrize('bad', ['extra', 'type', 'timestamp', 'hash', 'oversize'])
def test_malformed_snapshot_cannot_supply_launch_facts_and_errors_are_static(snapshot, bad):
    candidate = deepcopy(snapshot)
    if bad == 'extra':
        candidate[SECRET] = 'C:/' + SECRET
    elif bad == 'type':
        candidate['schema_version'] = True
    elif bad == 'timestamp':
        candidate['verified_at'] = SECRET
    elif bad == 'hash':
        candidate['decoder_sha256'] = SECRET
    else:
        candidate['run_id'] = SECRET * 2000
    with pytest.raises(ValueError) as error:
        validate_snapshot(candidate)
    assert SECRET not in str(error.value)
    value, status = decode_snapshot(json.dumps(candidate))
    assert value is None and status == 'invalid'


@pytest.mark.parametrize('ack_kind', ['missing', 'mismatch', 'invalid'])
def test_optional_ack_failure_keeps_transcript_but_never_claims_matching_runtime(config, snapshot, tmp_path, ack_kind):
    async def run():
        payload = reply(snapshot).json()
        if ack_kind == 'missing':
            payload.pop('runtime_snapshot')
        elif ack_kind == 'mismatch':
            payload['runtime_snapshot']['run_id'] = 'b' * 32
        else:
            payload['runtime_snapshot'][SECRET] = SECRET
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as client:
            stt = adapter(config, tmp_path, client)
            result = await stt.transcribe(bytes(1024))
            assert result.text == 'unchanged raw transcript' and len(stt.calls) == 1
            assert stt.calls[0]['runtime_provenance']['response_binding'] == ack_kind
            safe_calls(stt)
    asyncio.run(run())


@pytest.mark.parametrize('endpoint', ['https://remote.invalid/transcribe', LOCAL + '?claim=1',
    LOCAL + '#claim', 'http://user:password@127.0.0.1:8012/transcribe'])
def test_matching_remote_or_nonexact_url_claims_cannot_inherit_local_snapshot(config, snapshot, tmp_path, endpoint):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: reply(snapshot))) as client:
            stt = adapter(config, tmp_path, client, endpoint=endpoint)
            assert (await stt.transcribe(bytes(1024))).text == 'unchanged raw transcript'
            provenance = stt.calls[0]['runtime_provenance']
            assert provenance['response_binding'] == 'not_applicable'
            assert not provenance.get('launch_snapshot')
            assert snapshot['run_id'] not in json.dumps(provenance)
            safe_calls(stt)
    asyncio.run(run())


def test_redirect_to_remote_cannot_reuse_matching_ack_of_original_loopback_url(config, snapshot, tmp_path):
    async def run():
        seen = []
        def handle(request):
            seen.append(request.url.host)
            if request.url.host == '127.0.0.1':
                return httpx.Response(307, headers={'Location': 'https://remote.invalid/transcribe'})
            return reply(snapshot)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle), follow_redirects=True) as client:
            stt = adapter(config, tmp_path, client)
            assert (await stt.transcribe(bytes(1024))).text == 'unchanged raw transcript'
            assert seen == ['127.0.0.1', 'remote.invalid']
            provenance = stt.calls[0]['runtime_provenance']
            assert provenance['response_binding'] == 'not_applicable' and not provenance.get('launch_snapshot')
            safe_calls(stt)
    asyncio.run(run())


def test_fallback_failure_and_success_have_independent_runtime_evidence(config, snapshot, tmp_path):
    async def run():
        def handle(request):
            if request.url.host == '127.0.0.1':
                return httpx.Response(504, text=SECRET)
            return httpx.Response(200, json={'text': 'fallback transcript', 'segments': [], 'runtime_snapshot': runtime_ack(snapshot)})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            stt = adapter(config, tmp_path, client, fallback=True)
            stt.limiter.reserve = lambda _: None
            assert (await stt.transcribe(bytes(1024))).provider == 'groq'
            assert [call['provider'] for call in stt.calls] == ['moulsot', 'groq']
            assert stt.calls[0]['runtime_provenance']['response_binding'] == 'unavailable'
            other = stt.calls[1]['runtime_provenance']
            assert other['response_binding'] == 'not_applicable' and not other.get('launch_snapshot')
            assert snapshot['run_id'] not in json.dumps(other)
            safe_calls(stt)
    asyncio.run(run())


def test_cancelled_attempt_has_final_snapshot_without_claiming_response(config, tmp_path):
    async def run():
        entered = asyncio.Event()
        async def handle(request):
            entered.set()
            await asyncio.Event().wait()
        async with asyncio.timeout(3), httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            stt = adapter(config, tmp_path, client)
            task = asyncio.create_task(stt.transcribe(bytes(1024)))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            call, = stt.calls
            assert call['cancelled'] and call['elapsed_ms'] >= 0
            assert call['runtime_provenance']['response_binding'] == 'unavailable'
            safe_calls(stt)
    asyncio.run(run())


def test_response_binding_is_independent_of_transcription_success(config, snapshot, tmp_path):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: reply(snapshot, text=[SECRET]))) as client:
            stt = adapter(config, tmp_path, client)
            with pytest.raises(STTError) as error:
                await stt.transcribe(bytes(1024))
            assert SECRET not in str(error.value)
            call, = stt.calls
            assert not call['ok']
            # The bridge acknowledged its snapshot; this does not make invalid
            # text a successful recognition or attest what executed upstream.
            assert call['runtime_provenance']['response_binding'] == 'matched'
            safe_calls(stt)
    asyncio.run(run())


def test_bridge_and_adapter_capture_same_launch_once_without_rehashing_models(config, snapshot, tmp_path, monkeypatch, bridge):
    async def run():
        upstream_calls = []
        def handle(request):
            upstream_calls.append(request)
            return httpx.Response(200, json=upstream_result(bridge))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as upstream:
            app = bridge.create_app(client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
                stt = adapter(config, tmp_path, client)
                other = deepcopy(snapshot)
                other['run_id'] = 'b' * 32
                monkeypatch.setenv(ENV, json.dumps(other))
                monkeypatch.setattr(hashlib, 'file_digest', lambda *args, **kwargs: pytest.fail('Repeated requests must not hash model files'))
                for _ in range(2):
                    assert (await stt.transcribe(bytes(1024))).text == 'سلام لاباس'
                assert len(upstream_calls) == 2
                for call in stt.calls:
                    provenance = call['runtime_provenance']
                    assert provenance['response_binding'] == 'matched'
                    assert provenance['launch_snapshot']['run_id'] == snapshot['run_id']
                safe_calls(stt)
                assert not upstream.is_closed
    asyncio.run(run())


@pytest.mark.parametrize('environment_status', ['missing', 'invalid'])
def test_standalone_bridge_does_not_promote_upstream_runtime_claims_without_valid_startup_snapshot(config, snapshot, tmp_path, monkeypatch, bridge, environment_status):
    if environment_status == 'missing':
        monkeypatch.delenv(ENV)
    else:
        monkeypatch.setenv(ENV, SECRET)

    async def run():
        payload = upstream_result(bridge, runtime_snapshot=runtime_ack(snapshot),
            provenance={'verified': True, 'private_path': SECRET})
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as upstream:
            app = bridge.create_app(client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
                stt = adapter(config, tmp_path, client)
                assert (await stt.transcribe(bytes(1024))).text == 'سلام لاباس'
                provenance = stt.calls[0]['runtime_provenance']
                assert provenance['launch_snapshot_status'] == environment_status
                assert provenance['response_binding'] == 'missing'
                assert not provenance.get('launch_snapshot')
                assert snapshot['run_id'] not in json.dumps(provenance)
                safe_calls(stt)
    asyncio.run(run())


@pytest.mark.parametrize('verified', [False, True])
def test_supervisor_replaces_inherited_claim_only_after_verified_launch_and_never_sends_it_to_model(rig, snapshot, monkeypatch, verified):
    monkeypatch.setenv(ENV, json.dumps(snapshot))
    check = {'ready': True, 'issues': [], 'selected_device': 'cpu'}
    if verified:
        check.update(verification='launch_sha256_and_extracted_runtime', verified_at=snapshot['verified_at'])
    monkeypatch.setattr(rig.module, 'preflight', lambda **kwargs: deepcopy(check))
    rig.args.device = 'cpu'
    assert rig.run() == 0
    model, bridge_call, engine = rig.control.calls
    model_has_snapshot = ENV in model[2]['env']
    assert model_has_snapshot is False
    if verified:
        serialized = bridge_call[2]['env'][ENV]
        assert engine[2]['env'][ENV] == serialized
        current = validate_snapshot(json.loads(serialized))
        assert current['run_id'] != snapshot['run_id']
        assert current['selected_device'] == 'cpu' and current['verified_at'] == snapshot['verified_at']
    else:
        owned_has_snapshot = any(ENV in call[2]['env'] for call in (bridge_call, engine))
        assert owned_has_snapshot is False


def test_hosted_recovery_clears_previous_and_inherited_local_launch_claims(rig, snapshot, monkeypatch):
    monkeypatch.setenv(ENV, json.dumps(snapshot))
    monkeypatch.setattr(rig.module, 'preflight', lambda **kwargs: {'ready': True, 'issues': [],
        'selected_device': 'cuda', 'verification': 'launch_sha256_and_extracted_runtime',
        'verified_at': snapshot['verified_at']})
    rig.args.hosted_fallback = True
    rig.control.stop_after_ready = False
    rig.control.fail_local = rig.control.fail_hosted = True
    assert rig.run() == 1
    assert [call[0] for call in rig.control.calls] == ['model', 'bridge', 'engine', 'engine']
    local_has_snapshot = ENV in rig.control.calls[2][2]['env']
    assert local_has_snapshot is True
    hosted = rig.control.calls[-1][2]['env']
    hosted_has_snapshot = ENV in hosted
    assert hosted_has_snapshot is False
    assert hosted['MOULSOT_PROTOCOL'] == 'gradio'
    assert all(job.closed == 1 for job in rig.control.jobs)


def test_size_only_preflight_cannot_issue_hash_verification_stamp(rig, config, monkeypatch):
    # This exercises the real preflight's verify=False branch without assets.
    monkeypatch.setattr(rig.module, 'ASSETS', [])
    monkeypatch.setattr(rig.module, 'occupied_ports', lambda: [])
    monkeypatch.setattr(rig.module, 'digest', lambda *_: pytest.fail('verify=False must not hash files'))
    directory = rig.module.ROOT / 'configs'
    directory.mkdir()
    (directory / 'pizza.json').write_text(json.dumps({'runtime': config['runtime']}), encoding='utf-8')
    check = rig.real_preflight(verify=False, device='cpu')
    assert check['ready']
    assert check.get('verification') != 'launch_sha256_and_extracted_runtime'
    assert not check.get('verified_at')
    monkeypatch.setattr(rig.module, 'ASSETS', [('missing.gguf', 1, 'a' * 64, 'unused')])
    failed = rig.real_preflight(verify=True, device='cpu')
    assert not failed['ready']
    assert failed.get('verification') != 'launch_sha256_and_extracted_runtime'
    assert not failed.get('verified_at')
