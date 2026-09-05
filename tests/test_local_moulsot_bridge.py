"""Optional local bridge contracts, with in-memory HTTP only and no models."""

import asyncio
import base64
import importlib.util
import io
import json
from pathlib import Path
import wave

import httpx
import pytest


@pytest.fixture
def bridge():
    path = Path(__file__).resolve().parents[1] / 'deploy/local-moulsot/bridge.py'
    spec = importlib.util.spec_from_file_location('local_moulsot_bridge_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audio_bytes(*, rate=16000, channels=1, samples=512):
    output = io.BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        # Include CRLF inside binary sample data: multipart parsing must not
        # normalize audio bytes as though they were line-oriented text.
        audio.writeframes(b'\r\n' * (samples * channels))
    return output.getvalue()


def result(bridge, **changes):
    payload = {'model': bridge.EXPECTED_MODEL, 'choices': [{'finish_reason': 'stop',
        'message': {'role': 'assistant', 'content': 'language Arabic<asr_text>سلام لاباس'}}]}
    payload.update(changes)
    return payload


async def request_to(bridge, upstream_response=None, **kwargs):
    calls = []
    def handle(request):
        calls.append(request)
        return upstream_response if upstream_response is not None else httpx.Response(200, json=result(bridge))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as upstream:
        app = bridge.create_app(client=upstream)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://bridge.test') as client:
            response = await client.post('/transcribe', **kwargs)
        assert not upstream.is_closed
    return response, calls


def test_health_does_not_probe_or_allocate_an_upstream_request(bridge):
    async def run():
        def forbidden(request):
            pytest.fail('Health must not probe an inference server')
        async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as upstream:
            app = bridge.create_app(client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://bridge.test') as client:
                response = await client.get('/health')
                assert response.status_code == 200
                assert not upstream.is_closed
    asyncio.run(run())


@pytest.mark.parametrize('model', ['moulsot.v0.3.Q4_K_M.gguf', '/models/moulsot.v0.3.Q4_K_M.gguf',
                                  r'C:\models\moulsot.v0.3.Q4_K_M.gguf'])
def test_real_multipart_wav_returns_text_and_honest_quantized_provenance(bridge, model):
    raw = audio_bytes()
    reply, calls = asyncio.run(request_to(bridge, httpx.Response(200, json=result(bridge, model=model)),
        files={'file': ('capture.wav', raw, 'audio/wav')}))
    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert body['text'] == 'سلام لاباس' and body['confidence'] is None
    assert body['model_variant'] == 'moulsot.v0.3-Q4_K_M-mmproj-Q8_0'
    assert body['provenance']['source_model'] == 'atlasia/moulsot.v0.3'
    assert body['provenance']['quantized'] is True and body['provenance']['upstream_model'] == model
    assert len(calls) == 1 and str(calls[0].url) == 'http://127.0.0.1:8011/v1/chat/completions'
    submitted = json.loads(calls[0].content)
    audio_parts = [part['input_audio'] for message in submitted['messages']
                   for part in message['content'] if part.get('type') == 'input_audio']
    assert len(audio_parts) == 1 and audio_parts[0]['format'] == 'wav'
    assert base64.b64decode(audio_parts[0]['data']) == raw
    assert submitted['stream'] is False and submitted['cache_prompt'] is False
    assert submitted['max_tokens'] == bridge.MAX_OUTPUT_TOKENS == 512


def test_full_twenty_second_wav_and_long_completed_arabic_output(bridge):
    raw = audio_bytes(samples=20 * 16000)
    # A deliberately long mocked output, not a tokenizer or ASR-quality claim.
    text = ' '.join(['سلام', 'لاباس', 'بغيت', 'بيتزا'] * 40)
    assert len(text.split()) > 128
    data = result(bridge)
    data['choices'][0]['message']['content'] = bridge.PREFIX + text
    reply, calls = asyncio.run(request_to(bridge, httpx.Response(200, json=data),
        files={'file': ('twenty_seconds.wav', raw, 'audio/wav')}))
    assert reply.status_code == 200, reply.text
    assert reply.json()['text'] == text
    assert reply.json()['confidence'] is None
    assert len(calls) == 1
    submitted = json.loads(calls[0].content)
    assert submitted['max_tokens'] == bridge.MAX_OUTPUT_TOKENS == 512
    audio_parts = [part['input_audio'] for message in submitted['messages']
                   for part in message['content'] if part.get('type') == 'input_audio']
    assert len(audio_parts) == 1
    assert base64.b64decode(audio_parts[0]['data']) == raw


def test_prefix_only_result_is_empty_without_fabricated_confidence(bridge):
    data = result(bridge)
    data['choices'][0]['message']['content'] = 'language Arabic<asr_text>'
    reply, _ = asyncio.run(request_to(bridge, httpx.Response(200, json=data),
                                     files={'file': ('capture.wav', audio_bytes(), 'audio/wav')}))
    assert reply.status_code == 200 and reply.json()['text'] == ''
    assert reply.json()['confidence'] is None


@pytest.mark.parametrize('case,status', [('wrong_content_type', 415), ('not_wav', 400),
    ('stereo', 400), ('wrong_rate', 400), ('truncated', 400), ('too_long', 413),
    ('missing_file', 400), ('duplicate_file', 400), ('odd_data_size', 400), ('duplicate_data_chunk', 400)])
def test_malformed_audio_rejected_without_upstream_work(bridge, case, status):
    kwargs = {'files': {'file': ('capture.wav', audio_bytes(), 'audio/wav')}}
    if case == 'wrong_content_type':
        kwargs = {'content': b'raw audio', 'headers': {'Content-Type': 'application/octet-stream'}}
    elif case == 'missing_file':
        kwargs = {'files': {'other': ('capture.wav', audio_bytes(), 'audio/wav')}}
    elif case == 'duplicate_file':
        kwargs = {'files': [('file', ('a.wav', audio_bytes(), 'audio/wav')),
                           ('file', ('b.wav', audio_bytes(), 'audio/wav'))]}
    elif case in {'odd_data_size', 'duplicate_data_chunk'}:
        raw = bytearray(audio_bytes())
        if case == 'odd_data_size':
            raw.extend(b'\x01\x00')  # Odd declared PCM byte count plus proper RIFF padding.
            raw[40:44] = (1025).to_bytes(4, 'little')
        else:
            raw.extend(b'data\x02\x00\x00\x00\x01\x00')
        raw[4:8] = (len(raw) - 8).to_bytes(4, 'little')
        kwargs = {'files': {'file': ('capture.wav', bytes(raw), 'audio/wav')}}
    else:
        raw = {'not_wav': lambda: b'not a WAV', 'stereo': lambda: audio_bytes(channels=2),
               'wrong_rate': lambda: audio_bytes(rate=44100), 'truncated': lambda: audio_bytes()[:-2],
               'too_long': lambda: audio_bytes(samples=320001)}[case]()
        kwargs = {'files': {'file': ('capture.wav', raw, 'audio/wav')}}
    reply, calls = asyncio.run(request_to(bridge, **kwargs))
    assert reply.status_code == status, reply.text
    assert calls == []


def test_streamed_body_limit_is_enforced_without_content_length(bridge):
    async def run():
        async def chunks():
            for _ in range(bridge.MAX_BODY_BYTES // 32768 + 2):
                yield bytes(32768)
        reply, calls = await request_to(bridge, content=chunks(),
            headers={'Content-Type': 'multipart/form-data; boundary=offline'})
        assert reply.status_code == 413 and not calls
    asyncio.run(run())


@pytest.mark.parametrize('case', ['wrong_model', 'no_prefix', 'foreign_language', 'special_marker',
                                  'incomplete_marker', 'truncated_generation', 'missing_choice', 'oversized_response'])
def test_invalid_upstream_result_is_not_reported_as_transcription(bridge, case):
    data = result(bridge)
    if case == 'wrong_model':
        data['model'] = 'some-other-model.gguf'
    elif case == 'missing_choice':
        data['choices'] = []
    elif case == 'truncated_generation':
        data['choices'][0]['finish_reason'] = 'length'
    else:
        content = {'no_prefix': 'سلام', 'foreign_language': 'language English<asr_text>Hello',
                   'special_marker': 'language Arabic<asr_text>سلام<|im_end|>',
                   'incomplete_marker': 'language Arabic<asr_text>سلام<|im_end',
                   'oversized_response': 'language Arabic<asr_text>' + 'x' * (bridge.MAX_RESPONSE_BYTES + 1)}[case]
        data['choices'][0]['message']['content'] = content
    reply, calls = asyncio.run(request_to(bridge, httpx.Response(200, json=data),
        files={'file': ('capture.wav', audio_bytes(), 'audio/wav')}))
    assert reply.status_code == 502 and len(calls) == 1
    assert 'text' not in reply.json()


@pytest.mark.parametrize('failure,status', [('timeout', 504), ('http', 502)])
def test_upstream_failure_is_distinct_and_retry_after_failure_releases_busy(bridge, failure, status):
    async def run():
        count = 0
        def handle(request):
            nonlocal count
            count += 1
            if count == 1:
                if failure == 'timeout':
                    raise httpx.ReadTimeout('offline timeout', request=request)
                return httpx.Response(503)
            return httpx.Response(200, json=result(bridge))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as upstream:
            app = bridge.create_app(client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://bridge.test') as client:
                first = await client.post('/transcribe', files={'file': ('a.wav', audio_bytes(), 'audio/wav')})
                second = await client.post('/transcribe', files={'file': ('a.wav', audio_bytes(), 'audio/wav')})
                assert first.status_code == status and second.status_code == 200
                assert count == 2
    asyncio.run(run())


def test_concurrent_request_is_busy_and_cancellation_releases_reservation(bridge):
    async def run():
        entered, cancelled = asyncio.Event(), asyncio.Event()
        count = 0
        async def handle(request):
            nonlocal count
            count += 1
            if count == 1:
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            return httpx.Response(200, json=result(bridge))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as upstream:
            app = bridge.create_app(client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://bridge.test') as client:
                first = asyncio.create_task(client.post('/transcribe',
                    files={'file': ('a.wav', audio_bytes(), 'audio/wav')}))
                await asyncio.wait_for(entered.wait(), 1)
                second = await client.post('/transcribe', files={'file': ('b.wav', audio_bytes(), 'audio/wav')})
                assert second.status_code == 429 and count == 1
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                assert cancelled.is_set()
                third = await client.post('/transcribe', files={'file': ('c.wav', audio_bytes(), 'audio/wav')})
                assert third.status_code == 200 and count == 2
    asyncio.run(run())


def test_busy_reservation_starts_during_upload_and_cancelled_upload_never_calls_model(bridge):
    async def run():
        uploading = asyncio.Event()
        calls = []
        def handle(request):
            calls.append(request)
            return httpx.Response(200, json=result(bridge))
        async def partial_upload():
            yield b'--offline\r\n'
            uploading.set()
            await asyncio.Event().wait()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as upstream:
            app = bridge.create_app(client=upstream)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://bridge.test') as client:
                first = asyncio.create_task(client.post('/transcribe', content=partial_upload(),
                    headers={'Content-Type': 'multipart/form-data; boundary=offline'}))
                await asyncio.wait_for(uploading.wait(), 1)
                reply = await client.post('/transcribe', files={'file': ('a.wav', audio_bytes(), 'audio/wav')})
                assert reply.status_code == 429 and not calls
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                reply = await client.post('/transcribe', files={'file': ('a.wav', audio_bytes(), 'audio/wav')})
                assert reply.status_code == 200 and len(calls) == 1
    asyncio.run(run())


@pytest.mark.parametrize('injected', [False, True])
def test_lifespan_closes_only_owned_client_even_when_context_fails(bridge, monkeypatch, injected):
    async def run():
        real_client = httpx.AsyncClient
        created, options = [], []
        def forbidden(request):
            pytest.fail('Client lifecycle must not send inference')
        supplied = real_client(transport=httpx.MockTransport(forbidden)) if injected else None
        def construct(**kwargs):
            options.append(kwargs)
            client = real_client(transport=httpx.MockTransport(forbidden), **kwargs)
            created.append(client)
            return client
        monkeypatch.setattr(bridge.httpx, 'AsyncClient', construct)
        app = bridge.create_app(client=supplied)
        try:
            with pytest.raises(RuntimeError, match='offline shutdown'):
                async with app.router.lifespan_context(app):
                    if injected:
                        assert app.state.client is supplied and not created
                    else:
                        assert app.state.client is created[0]
                        assert options == [{'timeout': bridge.REQUEST_TIMEOUT_SECONDS,
                                            'trust_env': False, 'follow_redirects': False}]
                    raise RuntimeError('offline shutdown')
            if injected:
                assert not supplied.is_closed
            else:
                assert len(created) == 1 and created[0].is_closed
        finally:
            if supplied is not None:
                await supplied.aclose()
    asyncio.run(run())
