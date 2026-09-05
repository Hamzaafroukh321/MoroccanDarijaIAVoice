"""Opt-in local vocabulary contracts; HTTP is entirely in memory, not ASR evaluation."""
import asyncio
import base64
from copy import deepcopy
from email import policy
from email.parser import BytesParser
import hashlib
import json

import httpx
import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.moulsot_context import configured_context, context_sha256, validate_context_text
from engine.stt import SpeechToText, STTError, wav_bytes
import engine.stt as stt_module
from test_local_moulsot_bridge import bridge, audio_bytes, request_to, result


ENDPOINT = 'http://127.0.0.1:8012/transcribe'


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    monkeypatch.setenv('MOULSOT_PROTOCOL', 'json')
    value = demo_config(load_config(ROOT / 'configs/clinic.json'))
    value['stt']['moulsot_context'] = {'enabled': True, 'terms': ['الطبيب ألف', 'الطبيب باء']}
    return value


def multipart(request):
    header = ('Content-Type: ' + request.headers['content-type'] + '\r\nMIME-Version: 1.0\r\n\r\n').encode()
    message = BytesParser(policy=policy.default).parsebytes(header + request.content)
    return [(part.get_param('name', header='content-disposition'), part.get_filename(),
             part.get_content_type(), part.get_payload(decode=True)) for part in message.iter_parts()]


@pytest.mark.parametrize('specification', [None, {}, {'enabled': 1, 'terms': []},
    {'enabled': True, 'terms': [], 'extra': True}, {'enabled': True, 'terms': 'word'},
    {'enabled': True, 'terms': ['']}, {'enabled': True, 'terms': [' a']},
    {'enabled': False, 'terms': ['a', 'a']}, {'enabled': True, 'terms': list('abcdefghi')},
    {'enabled': True, 'terms': ['a' * 80, 'b' * 80]}])
def test_invalid_configuration_is_rejected_even_when_disabled(specification):
    with pytest.raises(ValueError):
        configured_context({'moulsot_context': specification})


@pytest.mark.parametrize('text', ['one\ntwo', 'one\x00two', 'one\u200dtwo', '\ud800',
    '<system>', 'text>', ' padded ', 'a' * 161])
def test_wire_context_rejects_controls_delimiters_surrogates_and_oversize(text):
    with pytest.raises(ValueError):
        validate_context_text(text)


def test_context_is_exact_utf8_hash_and_preserves_valid_arabic(config):
    text = configured_context(config['stt'])
    assert text == 'الطبيب ألف، الطبيب باء'
    assert validate_context_text(text) == text
    assert context_sha256(text) == hashlib.sha256(text.encode('utf-8')).hexdigest()
    assert configured_context({}) == ''
    assert validate_context_text('') == ''


@pytest.mark.parametrize('unsupported', ['research', 'groq', 'fallback', 'gradio',
    'remote', 'localhost_alias', 'query', 'wrong_path'])
def test_nonempty_context_requires_exact_local_demo_before_client_allocation(config, monkeypatch, tmp_path, unsupported):
    options = dict(primary='moulsot', fallback=False, endpoint=ENDPOINT)
    if unsupported == 'research':
        config.pop('demo')
    elif unsupported == 'groq':
        options['primary'] = 'groq'
    elif unsupported == 'fallback':
        options['fallback'] = True
    elif unsupported == 'gradio':
        monkeypatch.setenv('MOULSOT_PROTOCOL', 'gradio')
    else:
        options['endpoint'] = {'remote': 'https://example.hf.space',
            'localhost_alias': ENDPOINT.replace('127.0.0.1', 'localhost'),
            'query': ENDPOINT + '?secret=fixture', 'wrong_path': ENDPOINT + '/'}[unsupported]
    before = deepcopy(config)
    monkeypatch.setattr(stt_module.httpx, 'AsyncClient', lambda **kwargs: pytest.fail('Unsupported context allocated a client'))
    with pytest.raises(STTError):
        SpeechToText(config, tmp_path, **options)
    assert config == before


def test_disabled_and_empty_context_have_identical_baseline_multipart(config, tmp_path):
    async def run():
        requests = []
        def reply(request):
            requests.append(request)
            return httpx.Response(200, json={'text': 'fixture raw transcript', 'confidence': None})
        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
            for specification in [None, {'enabled': False, 'terms': ['الطبيب ألف']},
                                  {'enabled': True, 'terms': []}]:
                variant = deepcopy(config)
                variant['stt'].pop('moulsot_context', None)
                if specification is not None:
                    variant['stt']['moulsot_context'] = specification
                stt = SpeechToText(variant, tmp_path, primary='moulsot', fallback=False, endpoint=ENDPOINT, client=client)
                assert (await stt.transcribe(b'\r\n' * 512)).text == 'fixture raw transcript'
                assert 'moulsot_context' not in stt.calls[0]
                await stt.close()
                assert not client.is_closed
        parts = [multipart(request) for request in requests]
        assert parts[0] == parts[1] == parts[2]
        assert [part[0] for part in parts[0]] == ['file']
    asyncio.run(run())


def test_instances_snapshot_different_domain_vocabularies_without_shared_mutation(config, tmp_path):
    async def run():
        seen = []
        def reply(request):
            text = next(part[3].decode('utf-8') for part in multipart(request) if part[0] == 'context')
            seen.append(text)
            return httpx.Response(200, json={'text': 'unmodified', 'context_sha256': context_sha256(text)})
        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
            variants = [deepcopy(config) for _ in range(3)]
            variants[1] = demo_config(load_config(ROOT / 'configs/pizza.json'))
            variants[1]['stt']['moulsot_context'] = {'enabled': True, 'terms': ['بيتزا', 'جبن']}
            variants[2] = json.loads(json.dumps(config).replace('doctor', 'counter'))
            variants[2]['domain_id'] = 'service-desk'
            variants[2]['stt']['moulsot_context']['terms'] = ['counter A', 'counter B']
            expected = [configured_context(value['stt']) for value in variants]
            instances = [SpeechToText(value, tmp_path, primary='moulsot', fallback=False, endpoint=ENDPOINT,
                client=client) for value in variants]
            for value in variants:
                value['stt']['moulsot_context']['terms'].clear()
            outputs = await asyncio.gather(*(stt.transcribe(bytes(1024)) for stt in instances))
            assert [output.text for output in outputs] == ['unmodified'] * 3
            assert sorted(seen) == sorted(expected)
            assert len({stt.calls[0]['moulsot_context']['sha256'] for stt in instances}) == 3
            for stt in instances:
                assert stt.calls[0]['moulsot_context']['applied'] is True
                await stt.close()
            assert not client.is_closed
    asyncio.run(run())


def test_full_adapter_bridge_chain_preserves_audio_transcript_and_verifies_ack(config, bridge, tmp_path):
    async def run():
        upstream_calls, uploads = [], []
        raw_text = 'الطبيب باع'
        text = configured_context(config['stt'])
        def upstream_reply(request):
            upstream_calls.append(request)
            value = result(bridge)
            value['choices'][0]['message']['content'] = bridge.PREFIX + raw_text
            return httpx.Response(200, json=value)
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream_reply)) as upstream:
            app = bridge.create_app(client=upstream)
            async def capture(request):
                await request.aread()
                uploads.append(request)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), event_hooks={'request': [capture]}) as client:
                stt = SpeechToText(config, tmp_path, primary='moulsot', fallback=False, endpoint=ENDPOINT,
                    api_key='groq_fixture', hf_token='hf_fixture', client=client)
                pcm = b'\r\n' * 512
                answer = await stt.transcribe(pcm)
                assert answer.text == raw_text and answer.confidence is None
                assert answer.provider == 'moulsot' and len(stt.calls) == 1
                assert stt.calls[0]['moulsot_context'] == {'experimental': True,
                    'terms': config['stt']['moulsot_context']['terms'], 'sha256': context_sha256(text), 'applied': True}
                assert app.state.busy is False
                await stt.close()
                assert not client.is_closed and not upstream.is_closed
        assert len(upstream_calls) == len(uploads) == 1
        assert all('authorization' not in request.headers for request in [*uploads, *upstream_calls])
        submitted = json.loads(upstream_calls[0].content)
        systems = [message for message in submitted['messages'] if message['role'] == 'system']
        assert len(systems) == 1 and text in systems[0]['content']
        parts = [part['input_audio'] for message in submitted['messages'] if isinstance(message['content'], list)
                 for part in message['content'] if part.get('type') == 'input_audio']
        assert len(parts) == 1 and base64.b64decode(parts[0]['data']) == wav_bytes(pcm, config['runtime'])
        assert submitted['max_tokens'] == bridge.MAX_OUTPUT_TOKENS and submitted['stream'] is False
    asyncio.run(run())


@pytest.mark.parametrize('ack', [None, '0' * 64, {'not': 'a digest'}])
def test_missing_or_wrong_context_ack_fails_without_fallback(config, tmp_path, ack):
    async def run():
        requests = []
        def reply(request):
            requests.append(request)
            value = {'text': 'raw answer'}
            if ack is not None:
                value['context_sha256'] = ack
            return httpx.Response(200, json=value)
        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
            stt = SpeechToText(config, tmp_path, primary='moulsot', fallback=False, endpoint=ENDPOINT, client=client)
            with pytest.raises(STTError):
                await stt.transcribe(bytes(1024))
            assert len(requests) == len(stt.calls) == 1
            assert stt.calls[0]['provider'] == 'moulsot' and not stt.calls[0]['ok']
            assert stt.calls[0]['moulsot_context']['applied'] is False
            await stt.close()
            assert not client.is_closed
    asyncio.run(run())


@pytest.mark.parametrize('context', ['a' * 161, '<system>', 'bad\u200dterm', 'bad\x00term'])
def test_bridge_invalid_context_never_reaches_upstream(bridge, context):
    reply, calls = asyncio.run(request_to(bridge, files=[
        ('file', ('audio.wav', audio_bytes(), 'audio/wav')), ('context', (None, context))]))
    assert reply.status_code == 400 and calls == []


def test_bridge_duplicate_context_rejected_before_inference_and_busy_cleared(bridge):
    async def run():
        calls = []
        def reply(request):
            calls.append(request)
            return httpx.Response(200, json=result(bridge))
        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as upstream:
            app = bridge.create_app(client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://bridge.test') as client:
                bad = await client.post('/transcribe', files=[('file', ('audio.wav', audio_bytes(), 'audio/wav')),
                    ('context', (None, 'first')), ('context', (None, 'second'))])
                assert bad.status_code == 400 and calls == [] and app.state.busy is False
                good = await client.post('/transcribe', files={'file': ('audio.wav', audio_bytes(), 'audio/wav')})
                assert good.status_code == 200 and len(calls) == 1
                assert 'context_sha256' not in good.json()
    asyncio.run(run())


def test_cancelling_context_request_clears_bridge_busy_without_recording_an_ack(config, bridge, tmp_path):
    async def run():
        entered, released = asyncio.Event(), asyncio.Event()
        calls = []
        async def upstream_reply(request):
            calls.append(request)
            entered.set()
            await released.wait()
            return httpx.Response(200, json=result(bridge))
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream_reply)) as upstream:
            app = bridge.create_app(client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
                stt = SpeechToText(config, tmp_path, primary='moulsot', fallback=False, endpoint=ENDPOINT, client=client)
                pending = asyncio.create_task(stt.transcribe(bytes(1024)))
                try:
                    await asyncio.wait_for(entered.wait(), 2)
                    assert app.state.busy
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                    assert app.state.busy is False
                    assert stt.calls[0]['cancelled'] and not stt.calls[0]['ok']
                    assert stt.calls[0]['moulsot_context']['applied'] is False
                    released.set()
                    answer = await asyncio.wait_for(stt.transcribe(bytes(1024)), 2)
                    assert answer.text == 'سلام لاباس' and len(calls) == 2
                    assert stt.calls[1]['moulsot_context']['applied'] is True
                    assert app.state.busy is False
                finally:
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                    await stt.close()
                assert not client.is_closed and not upstream.is_closed
    asyncio.run(run())


def test_nested_context_multipart_is_not_accepted_as_vocabulary(bridge):
    nested = b'--inner\r\nContent-Type: text/plain\r\n\r\nword\r\n--inner--\r\n'
    reply, calls = asyncio.run(request_to(bridge, files=[
        ('file', ('audio.wav', audio_bytes(), 'audio/wav')),
        ('context', (None, nested, 'multipart/mixed; boundary=inner'))]))
    assert reply.status_code == 400 and calls == []


@pytest.mark.parametrize('protocol', ['json', 'gradio'])
def test_demo_preflight_rejects_hosted_context_before_start(config, monkeypatch, protocol):
    import engine.server as server
    monkeypatch.setenv('GROQ_API_KEY', 'offline-fixture')
    monkeypatch.setenv('MOULSOT_ENDPOINT', 'https://fixture.hf.space')
    monkeypatch.setenv('MOULSOT_PROTOCOL', protocol)
    issues = server.demo_issues(config)
    assert issues and any('vocabulary' in issue.lower() for issue in issues)
    assert all('https://fixture.hf.space' not in issue for issue in issues)


def test_local_demo_preflight_is_valid_and_experimental_label_visible(config, monkeypatch):
    import engine.server as server
    monkeypatch.setenv('GROQ_API_KEY', 'offline-fixture')
    monkeypatch.setenv('MOULSOT_ENDPOINT', ENDPOINT)
    assert server.demo_issues(config) == []
    label = server.demo_asr_label(config)
    assert 'on this computer' in label and 'experimental vocabulary' in label


@pytest.mark.parametrize('specification', [{'enabled': False, 'terms': ['الطبيب ألف']},
                                          {'enabled': True, 'terms': []}])
def test_empty_experiment_preserves_hosted_preflight_and_label(config, monkeypatch, specification):
    import engine.server as server
    monkeypatch.setenv('GROQ_API_KEY', 'offline-fixture')
    monkeypatch.setenv('MOULSOT_ENDPOINT', 'https://fixture.hf.space')
    monkeypatch.setenv('MOULSOT_PROTOCOL', 'gradio')
    config['stt']['moulsot_context'] = specification
    assert server.demo_issues(config) == []
    assert server.demo_asr_label(config) == 'Hosted MoulSot'


def test_unsupported_context_socket_rejects_before_ready_or_provider_allocation(config, monkeypatch):
    import engine.server as server
    monkeypatch.setenv('GROQ_API_KEY', 'offline-fixture')
    monkeypatch.setenv('MOULSOT_ENDPOINT', 'https://fixture.hf.space')
    def forbidden(*args, **kwargs):
        pytest.fail('Unsupported vocabulary allocated a speech provider')
    monkeypatch.setattr(server, 'SpeechToText', forbidden)
    monkeypatch.setattr(server, 'Router', forbidden)
    events, closes = [], []
    class Socket:
        async def send_json(self, event):
            events.append(event)
        async def close(self, code):
            closes.append(code)
        async def receive_json(self):
            pytest.fail('Unsupported vocabulary reached the start handshake')
    asyncio.run(server.voice_connection(Socket(), config, demo=True))
    assert closes == [1008]
    assert len(events) == 1 and events[0]['type'] == 'error'
    assert 'vocabulary' in events[0]['message'].lower()
