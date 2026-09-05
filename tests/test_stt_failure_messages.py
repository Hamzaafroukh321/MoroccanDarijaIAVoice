"""Safe, actionable ASR failure categories with in-memory HTTP only."""
import asyncio
from contextlib import contextmanager
import json

import httpx
import pytest

from engine.config import ROOT, load_config
import engine.stt as stt_module
from engine.stt import MoulSotQuotaError, SpeechToText, STTError, RateLimitError


@pytest.fixture
def config(monkeypatch):
    monkeypatch.delenv('MOULSOT_PROTOCOL', raising=False)
    result = load_config(ROOT / 'configs/pizza.json')
    result['stt']['moulsot_protocol'] = 'json'
    return result


def check_private(error, calls):
    serialized = str(error) + json.dumps(calls)
    for private in ('private-body', 'private-password', 'private-query', 'private-header',
                    'private-message', 'private-key', 'private-audio', 'https://', 'http://'):
        assert private not in serialized


@pytest.mark.parametrize('status,kind,phrase', [
    (429, 'busy_or_rate_limited', 'busy or rate-limited'),
    (504, 'timeout', 'Local MoulSot recognition timed out'),
    (502, 'service_failure', 'service failed'),
    (503, 'service_failure', 'service failed'),
    (401, 'authentication', 'access was denied'),
    (403, 'authentication', 'access was denied'),
    (400, 'request_rejected', 'request was rejected')])
def test_http_failures_have_safe_status_and_actionable_message(config, tmp_path, status, kind, phrase):
    def handle(request):
        return httpx.Response(status, text='private-body transcript and credentials',
            headers={'Retry-After': 'private-header'})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            stt = SpeechToText(config, tmp_path, primary='moulsot', fallback=False, client=client,
                endpoint='http://user:private-password@127.0.0.1:8012/private-audio?token=private-query')
            with pytest.raises(STTError, match=phrase) as caught:
                await stt.transcribe(bytes(1024))
            record, = stt.calls
            assert record['error'] == 'HTTPStatusError'
            assert record['http_status'] == status and record['failure_kind'] == kind
            assert record['phases_ms']['json_request'] >= 0
            if status in {429, 504, 502, 503}:
                assert 'quota' not in str(caught.value).lower()
                assert 'slow' not in str(caught.value).lower()
            check_private(caught.value, stt.calls)
    asyncio.run(run())


@pytest.mark.parametrize('endpoint,local', [('http://localhost:8012/transcribe', True),
    ('http://[::1]:8012/transcribe', True), ('https://private-host.invalid/transcribe', False)])
def test_network_timeout_labels_only_loopback_json_as_local(config, tmp_path, endpoint, local):
    def handle(request):
        raise httpx.ReadTimeout('private-message private-query', request=request)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            stt = SpeechToText(config, tmp_path, primary='moulsot', fallback=False,
                endpoint=endpoint, client=client)
            with pytest.raises(STTError, match='recognition timed out') as caught:
                await stt.transcribe(bytes(1024))
            assert str(caught.value).startswith('Local MoulSot' if local else 'MoulSot')
            assert stt.calls[0]['failure_kind'] == 'timeout'
            assert stt.calls[0]['error'] == 'ReadTimeout'
            assert 'http_status' not in stt.calls[0]
            check_private(caught.value, stt.calls)
    asyncio.run(run())


def test_overall_timeout_keeps_timing_and_does_not_blame_speaker(config, tmp_path):
    config['stt']['timeout_ms'] = 5
    async def handle(request):
        await asyncio.Event().wait()
    async def run():
        async with asyncio.timeout(2), httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            stt = SpeechToText(config, tmp_path, primary='moulsot', fallback=False,
                endpoint='http://127.0.0.1:8012/transcribe', client=client)
            with pytest.raises(STTError, match='Local MoulSot recognition timed out') as caught:
                await stt.transcribe(bytes(1024))
            assert stt.calls[0]['error'] == 'TimeoutError'
            assert stt.calls[0]['failure_kind'] == 'timeout'
            assert stt.calls[0]['elapsed_ms'] >= stt.calls[0]['phases_ms']['json_request']
            assert 'shorter' not in str(caught.value) and 'quota' not in str(caught.value)
    asyncio.run(run())


@pytest.mark.parametrize('fallback_status', [200, 401])
def test_fallback_success_or_last_failure_takes_precedence(config, tmp_path, fallback_status):
    def handle(request):
        if request.url.host == '127.0.0.1':
            return httpx.Response(504, text='private-body')
        return httpx.Response(fallback_status, json={'text': 'fallback result', 'segments': []},
            headers={'x-secret': 'private-header'})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            stt = SpeechToText(config, tmp_path, primary='moulsot', fallback=True,
                endpoint='http://127.0.0.1:8012/transcribe', api_key='private-key', client=client)
            stt.limiter.reserve = lambda _: None
            if fallback_status == 200:
                result = await stt.transcribe(bytes(1024))
                assert result.provider == 'groq' and result.text == 'fallback result'
                assert stt.calls[-1]['ok']
            else:
                with pytest.raises(STTError, match='Groq recognition access was denied') as caught:
                    await stt.transcribe(bytes(1024))
                assert 'timeout' not in str(caught.value)
                assert stt.calls[-1]['failure_kind'] == 'authentication'
                check_private(caught.value, stt.calls)
            assert [record['provider'] for record in stt.calls] == ['moulsot', 'groq']
            assert stt.calls[0]['failure_kind'] == 'timeout'
    asyncio.run(run())


def test_known_zerogpu_quota_keeps_existing_priority_and_safe_text(config, tmp_path):
    config['stt']['moulsot_protocol'] = 'gradio'
    def handle(request):
        if request.url.host != 'private-host.hf.space':
            return httpx.Response(401, text='private-body')
        if request.url.path.endswith('/upload'):
            return httpx.Response(200, json=['/tmp/private-audio'])
        if request.method == 'POST':
            return httpx.Response(200, json={'event_id': 'abc123'})
        return httpx.Response(200, text='event: error\ndata: ' + json.dumps({'error':
            'ZeroGPU runs limit exceeded. private-body private-key'}) + '\n\n')
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            stt = SpeechToText(config, tmp_path, primary='moulsot', fallback=True,
                endpoint='https://private-host.hf.space', hf_token='private-key', api_key='private-key', client=client)
            stt.limiter.reserve = lambda _: None
            with pytest.raises(MoulSotQuotaError, match='authenticated account') as caught:
                await stt.transcribe(bytes(1024))
            assert stt.calls[0]['failure_kind'] == 'quota'
            assert stt.calls[1]['failure_kind'] == 'authentication'
            check_private(caught.value, stt.calls)
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['unreadable', 'busy', 'request_budget', 'audio_budget'])
def test_limiter_availability_is_distinct_from_verified_exhaustion(config, tmp_path, monkeypatch, failure):
    if failure == 'request_budget':
        config['stt']['limits'] = [{'window_seconds': 60, 'requests': 0}]
    elif failure == 'audio_budget':
        config['stt']['limits'] = [{'window_seconds': 60, 'audio_seconds': 0}]
    if failure == 'busy':
        @contextmanager
        def busy_lock(path):
            # Even arbitrary text mentioning exhaustion must not set category.
            raise RateLimitError('private-message budget exhausted private-key')
            yield
        monkeypatch.setattr(stt_module, 'locked_file', busy_lock)
    def forbidden(request):
        pytest.fail('A failed local budget check must not send provider requests')
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
            stt = SpeechToText(config, tmp_path, primary='groq', client=client, api_key='private-key')
            if failure == 'unreadable':
                stt.limiter.path.parent.mkdir(parents=True, exist_ok=True)
                stt.limiter.path.write_text('private-body invalid JSON', encoding='utf-8')
            with pytest.raises(STTError) as caught:
                await stt.transcribe(bytes(1024))
            record, = stt.calls
            if failure.endswith('_budget'):
                assert record['error'] == 'BudgetExhaustedError'
                assert record['failure_kind'] == 'local_budget'
                assert 'budget is exhausted' in str(caught.value)
            else:
                assert record['error'] == 'RateLimitError'
                assert record['failure_kind'] == 'limiter_unavailable'
                assert 'could not be checked' in str(caught.value)
                assert 'exhausted' not in str(caught.value) and 'reset' not in str(caught.value)
            check_private(caught.value, stt.calls)
    asyncio.run(run())
