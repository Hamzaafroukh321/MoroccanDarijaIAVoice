import asyncio
import json

import httpx
import pytest

from engine.config import ROOT, load_config
from engine.demo import DemoVoice, demo_config, pcm16
from engine.responder import AudioBankError
from engine.state import Action
from engine.stt import wav_bytes


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    monkeypatch.setenv('HF_TOKEN', 'hf_fixture')
    monkeypatch.setenv('GROQ_API_KEY', 'do_not_send_groq')
    return demo_config(load_config(ROOT/'configs/pizza.json'))


def transport(demo, seen, event=None, audio=None):
    def handle(request):
        seen.append(request)
        assert request.url.host == 'medmac01-darija-arabic-tts.hf.space'
        assert request.headers['Authorization'] == 'Bearer hf_fixture'
        assert 'xi-api-key' not in request.headers
        if request.url.path == '/config':
            return httpx.Response(200, json={'components': [{'type': 'audio', 'props': {
                'label': 'Speaker reference', 'value': {'path': '/tmp/gradio/ref/speaker.wav'}}}]})
        if request.method == 'POST':
            data = json.loads(request.content)['data']
            assert data[1]['path'] == '/tmp/gradio/ref/speaker.wav'
            assert data[2] == demo['demo']['tts_temperature']
            return httpx.Response(200, json={'event_id': 'abc123'})
        if request.url.path.endswith('/abc123'):
            body = event if event is not None else ('event: heartbeat\ndata: null\n\nevent: complete\ndata: '
                + json.dumps([{'path': '/tmp/gradio/result/audio.wav', 'url': 'https://untrusted.invalid/audio.wav'}]) + '\n\n')
            return httpx.Response(200, text=body)
        assert request.url.path == '/gradio_api/file=/tmp/gradio/result/audio.wav'
        return httpx.Response(200, content=audio if audio is not None else
            wav_bytes(bytes(4800), dict(demo['runtime'], sample_rate_hz=24000)))
    return httpx.MockTransport(handle)


def test_xtts_converts_audio_caches_and_uses_only_hf_token(demo, tmp_path):
    async def run():
        seen = []
        async with httpx.AsyncClient(transport=transport(demo, seen)) as client:
            bank = DemoVoice(demo, tmp_path, client=client)
            for _ in range(2):
                rendered = await bank.render(Action('greeting'), {})
                assert rendered.total_samples == rendered.synthetic_samples == 1600
                assert len(pcm16(rendered.wav, demo['runtime'])) == 3200
            assert len(seen) == 4
            # Changed synthesis parameters must not reuse a previous voice cache entry.
            bank.settings['tts_temperature'] = .7
            await bank.render(Action('greeting'), {})
            assert len(seen) == 7
        assert not (tmp_path/demo['stt']['rate_limit_file']).exists()
    asyncio.run(run())


@pytest.mark.parametrize('event,match', [
    ('event: error\ndata: "ZeroGPU quota exceeded private detail"\n\n', 'quota'),
    ('event: error\ndata: "private internal failure"\n\n', 'generation failed'),
    ('event: complete\ndata: []\n\n', 'unexpected response'),
    ('event: complete\ndata: [{"path":"https://untrusted.invalid/audio.wav"}]\n\n', 'unexpected response'),
    ('event: heartbeat\ndata: null\n\n', 'without audio'),
])
def test_xtts_failure_is_sanitized_and_not_cached(demo, tmp_path, event, match):
    async def run():
        seen = []
        async with httpx.AsyncClient(transport=transport(demo, seen, event=event)) as client:
            bank = DemoVoice(demo, tmp_path, client=client)
            with pytest.raises(AudioBankError, match=match) as error:
                await bank.render(Action('greeting'), {})
            assert 'private' not in str(error.value)
        assert len(seen) == 3
        assert not list(tmp_path.rglob('*.wav'))
    asyncio.run(run())


def test_xtts_rejects_invalid_audio(demo, tmp_path):
    async def run():
        async with httpx.AsyncClient(transport=transport(demo, [], audio=b'not audio')) as client:
            with pytest.raises(AudioBankError, match='invalid audio'):
                await DemoVoice(demo, tmp_path, client=client).render(Action('greeting'), {})
        assert not list(tmp_path.rglob('*.wav'))
    asyncio.run(run())


def test_xtts_has_total_timeout(demo, tmp_path):
    async def handle(request):
        await asyncio.sleep(1)
        return httpx.Response(200)
    async def run():
        demo['demo']['tts_timeout_ms'] = 10
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with pytest.raises(AudioBankError, match='timed out'):
                await DemoVoice(demo, tmp_path, client=client).render(Action('greeting'), {})
    asyncio.run(run())
