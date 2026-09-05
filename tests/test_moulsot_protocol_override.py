"""Optional protocol override keeps caller configuration and credential scope intact."""

import asyncio
from copy import deepcopy

import httpx
import pytest

from engine.config import ROOT, load_config
import engine.stt as stt_module
from engine.stt import SpeechToText, STTError


@pytest.mark.parametrize('configured', ['json', 'gradio'])
@pytest.mark.parametrize('environment', [None, '', '   '])
def test_unset_or_blank_override_keeps_configured_protocol(monkeypatch, tmp_path, configured, environment):
    if environment is None:
        monkeypatch.delenv('MOULSOT_PROTOCOL', raising=False)
    else:
        monkeypatch.setenv('MOULSOT_PROTOCOL', environment)
    config = load_config(ROOT / 'configs/pizza.json')
    config['stt']['moulsot_protocol'] = configured
    before = deepcopy(config)
    stt = SpeechToText(config, tmp_path, client=object(), primary='moulsot', fallback=False)
    assert stt.settings['moulsot_protocol'] == configured
    assert stt.settings is not config['stt'] and config == before
    assert stt.primary == 'moulsot' and stt.fallback is False


def test_json_override_uses_exact_endpoint_without_forwarding_credentials_or_mutating_config(monkeypatch, tmp_path):
    monkeypatch.setenv('MOULSOT_PROTOCOL', ' json ')
    config = load_config(ROOT / 'configs/pizza.json')
    config['stt']['moulsot_protocol'] = 'gradio'
    before = deepcopy(config)
    requests = []
    def handle(request):
        requests.append(request)
        assert str(request.url) == 'http://127.0.0.1:8012/transcribe'
        assert 'authorization' not in request.headers
        return httpx.Response(200, json={'text': 'fixture answer', 'confidence': None})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            stt = SpeechToText(config, tmp_path, primary='moulsot', fallback=False,
                endpoint='http://127.0.0.1:8012/transcribe', api_key='groq_fixture',
                hf_token='hf_fixture', client=client)
            # The selected protocol belongs to this instance, not future env changes.
            monkeypatch.setenv('MOULSOT_PROTOCOL', 'gradio')
            result = await stt.transcribe(bytes(1024))
            assert result.provider == 'moulsot' and result.confidence is None
            assert stt.settings['moulsot_protocol'] == 'json'
            assert stt.primary == 'moulsot' and stt.fallback is False
            assert set(stt.calls[0]['phases_ms']) == {'json_request'}
            assert config == before and len(requests) == 1
    asyncio.run(run())


@pytest.mark.parametrize('endpoint,authenticated', [
    ('https://fixture.hf.space', True), ('http://fixture.hf.space', False),
    ('https://custom.invalid', False)])
def test_gradio_override_preserves_existing_hugging_face_credential_scope(monkeypatch, tmp_path, endpoint, authenticated):
    monkeypatch.setenv('MOULSOT_PROTOCOL', 'gradio')
    config = load_config(ROOT / 'configs/pizza.json')
    config['stt']['moulsot_protocol'] = 'json'
    requests = []
    def handle(request):
        requests.append(request)
        assert request.headers.get('Authorization') == ('Bearer hf_fixture' if authenticated else None)
        if request.url.path.endswith('/upload'):
            return httpx.Response(200, json=['/tmp/fixture.wav'])
        if request.method == 'POST':
            return httpx.Response(200, json={'event_id': 'fixture123'})
        return httpx.Response(200, text='event: complete\ndata: ["fixture answer"]\n\n')
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            stt = SpeechToText(config, tmp_path, primary='moulsot', fallback=False,
                endpoint=endpoint, hf_token='hf_fixture', api_key='groq_fixture', client=client)
            assert (await stt.transcribe(bytes(1024))).provider == 'moulsot'
            assert len(requests) == 3 and config['stt']['moulsot_protocol'] == 'json'
            assert set(stt.calls[0]['phases_ms']) == {'upload', 'submission', 'result_wait'}
    asyncio.run(run())


@pytest.mark.parametrize('invalid', ['unsupported', 'JSON', 'json|gradio'])
def test_unknown_override_fails_before_client_allocation(monkeypatch, tmp_path, invalid):
    monkeypatch.setenv('MOULSOT_PROTOCOL', invalid)
    config = load_config(ROOT / 'configs/pizza.json')
    before = deepcopy(config)
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid protocol allocated a client')
    monkeypatch.setattr(stt_module.httpx, 'AsyncClient', forbidden)
    with pytest.raises(STTError, match='MOULSOT_PROTOCOL.*json.*gradio'):
        SpeechToText(config, tmp_path)
    assert config == before


@pytest.mark.parametrize('primary,fallback,expected_calls', [('groq', False, 1), ('moulsot', True, 2)])
def test_override_does_not_change_primary_or_fallback_policy(monkeypatch, tmp_path, primary, fallback, expected_calls):
    monkeypatch.setenv('MOULSOT_PROTOCOL', 'json')
    config = load_config(ROOT / 'configs/pizza.json')
    requests = []
    def handle(request):
        requests.append(request)
        if request.url.host == '127.0.0.1':
            assert 'authorization' not in request.headers
            return httpx.Response(503)
        assert request.headers['Authorization'] == 'Bearer groq_fixture'
        return httpx.Response(200, json={'text': 'fallback fixture', 'segments': []})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            stt = SpeechToText(config, tmp_path, primary=primary, fallback=fallback,
                endpoint='http://127.0.0.1:8012/transcribe', api_key='groq_fixture',
                hf_token='hf_fixture', client=client)
            stt.limiter.reserve = lambda *args: None
            assert (await stt.transcribe(bytes(1024))).provider == 'groq'
            assert stt.primary == primary and stt.fallback is fallback
            assert len(requests) == expected_calls
    asyncio.run(run())


def test_missing_endpoint_guidance_identifies_isolated_service_and_direct_runtime_limit(monkeypatch, tmp_path):
    monkeypatch.delenv('MOULSOT_PROTOCOL', raising=False)
    config = load_config(ROOT / 'configs/pizza.json')
    stt = SpeechToText(config, tmp_path, endpoint='', client=object(), primary='moulsot', fallback=False)
    with pytest.raises(STTError, match='isolated local JSON service.*MOULSOT_PROTOCOL=json.*pinned runtime'):
        asyncio.run(stt._moulsot(b'fixture'))
