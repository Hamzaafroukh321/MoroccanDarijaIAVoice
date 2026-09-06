"""Known single-Textbox callback errors are service failures, not speech."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.endpointing import Segment
from engine.pipeline import VoiceSession
from engine.responder import RenderedAudio
from engine.stt import SpeechToText, STTError, wav_bytes


PRIVATE = 'PRIVATE_STATUS_SENTINEL'
ENDPOINT = 'https://owned-fixture.hf.space'
STATUSES = [
    "<div style='color:#666'>Please upload an audio file first.</div>",
    "<div style='color:red'>Unexpected result format.</div>",
    f"<div style='color:red'>Error: quota timeout {PRIVATE}</div>",
    f'''<div style='color:red'>Error: "quoted" '{PRIVATE}' C:\\private\\file
    second line بالدارجة https://private.invalid/?token={PRIVATE}</div>''',
]


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv('MOULSOT_PROTOCOL', 'gradio')
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    return demo_config(load_config(ROOT / 'configs/clinic.json'))


class Gradio:
    def __init__(self, data):
        self.data, self.requests = data, []

    def handle(self, request):
        self.requests.append(request)
        if request.url.host != 'owned-fixture.hf.space':
            return httpx.Response(200, json={'text': 'explicit fallback result', 'segments': []})
        if request.url.path.endswith('/upload'):
            return httpx.Response(200, json=['/tmp/private-input.wav'])
        if request.method == 'POST':
            return httpx.Response(200, json={'event_id': 'job123'})
        return httpx.Response(200, text='event: heartbeat\ndata: null\n\nevent: complete\ndata: '
            + json.dumps(self.data) + '\n\n')


def adapter(config, root, client, *, fallback=False):
    return SpeechToText(config, root, primary='moulsot', endpoint=ENDPOINT,
        api_key='offline-key', hf_token='offline-hf', fallback=fallback, client=client)


def assert_safe(value):
    serialized = value if isinstance(value, str) else json.dumps(value)
    for private in (PRIVATE, 'private-input', 'private.invalid', '<div', 'color:red', 'quoted'):
        assert private not in serialized


@pytest.mark.parametrize('status', STATUSES, ids=['missing-upload', 'unexpected-format', 'arbitrary-cause', 'quotes-unicode-controls'])
def test_source_verified_single_output_failure_is_rejected_without_reflecting_error_text(config, tmp_path, status):
    async def run():
        # Python repr here reproduces the verified Textbox conversion. The
        # production guard must not eval, decode or interpret this error body.
        provider = Gradio([repr(('', '', None, status))])
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider.handle)) as client:
            stt = adapter(config, tmp_path, client)
            stt.limiter.reserve = lambda *_: pytest.fail('Disabled fallback must not use Groq quota')
            with pytest.raises(STTError) as error:
                await stt.transcribe(bytes(1024))
            assert_safe(str(error.value))
            assert 'slow' not in str(error.value).lower() and 'quota' not in str(error.value).lower()
            assert len(provider.requests) == 3
            assert all(request.url.host == 'owned-fixture.hf.space' for request in provider.requests)
            call, = stt.calls
            assert not call['ok'] and call['provider'] == 'moulsot'
            assert call['failure_kind'] == 'invalid_response_or_configuration'
            assert set(call['phases_ms']) == {'upload', 'submission', 'result_wait'}
            assert_safe(stt.calls)
    asyncio.run(run())


@pytest.mark.parametrize('data,expected', [
    ([''], ''),
    (['  '], ''),
    (['The word Error: is part of a normal transcript.'], 'The word Error: is part of a normal transcript.'),
    ([repr(('', '', None, '<div>Ordinary status</div>'))], repr(('', '', None, '<div>Ordinary status</div>'))),
    (['valid transcript', '', None, STATUSES[1]], 'valid transcript'),
    (['', '', None, STATUSES[1]], ''),
    ([repr(('nonempty', '', None, STATUSES[1]))], repr(('nonempty', '', None, STATUSES[1]))),
    ([repr(('', '', None, '<div style="color:red">Unknown failure wording.</div>'))],
     repr(('', '', None, '<div style="color:red">Unknown failure wording.</div>'))),
])
def test_empty_and_unrelated_outputs_keep_existing_transcription_behavior(config, tmp_path, data, expected):
    async def run():
        provider = Gradio(data)
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider.handle)) as client:
            stt = adapter(config, tmp_path, client)
            assert (await stt.transcribe(bytes(1024))).text == expected
            assert len(stt.calls) == 1 and stt.calls[0]['ok']
            assert len(provider.requests) == 3
    asyncio.run(run())


@pytest.mark.parametrize('data', [[], {'text': 'wrong envelope'}, [None]])
def test_malformed_complete_shape_is_still_rejected(config, tmp_path, data):
    async def run():
        provider = Gradio(data)
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider.handle)) as client:
            stt = adapter(config, tmp_path, client)
            with pytest.raises(STTError):
                await stt.transcribe(bytes(1024))
            assert len(stt.calls) == 1 and not stt.calls[0]['ok']
    asyncio.run(run())


def test_explicitly_enabled_fallback_retains_its_existing_policy(config, tmp_path):
    async def run():
        provider = Gradio([repr(('', '', None, STATUSES[1]))])
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider.handle)) as client:
            stt = adapter(config, tmp_path, client, fallback=True)
            reserves = []
            stt.limiter.reserve = lambda _: reserves.append(True)
            output = await stt.transcribe(bytes(1024))
            assert output.provider == 'groq' and output.text == 'explicit fallback result'
            assert [call['ok'] for call in stt.calls] == [False, True]
            assert [call['provider'] for call in stt.calls] == ['moulsot', 'groq']
            assert len(reserves) == 1 and len(provider.requests) == 4
    asyncio.run(run())


@pytest.mark.parametrize('status', [STATUSES[1], STATUSES[3]], ids=['unexpected-format', 'quoted-error'])
def test_demo_stops_on_known_error_without_router_or_spoken_repair_and_preserves_pending_work(config, tmp_path, status):
    async def run():
        provider = Gradio([repr(('', '', None, status))])
        events, renders = [], []
        class Router:
            calls = []
            async def route(self, *args):
                pytest.fail('A hosted service failure must not reach the task router')
        class Bank:
            async def render(self, action, values):
                renders.append(action.kind)
                return RenderedAudio(wav_bytes(bytes(1024), config['runtime']), 512, 512, ['fixture'], 'Offline initial prompt')
        async def emit(event):
            events.append(deepcopy(event))
            if event['type'] == 'audio_end':
                await voice.playback_finished(event['playback_id'])
        async def audio(data):
            pass
        async with httpx.AsyncClient(transport=httpx.MockTransport(provider.handle)) as client:
            stt = adapter(config, tmp_path, client)
            voice = VoiceSession(config, tmp_path, stt, Router(), Bank(), None, emit, audio)
            voice.task.apply([{'op': 'set', 'slot': 'doctor', 'value': 'doctor_a'},
                {'op': 'set', 'slot': 'date', 'value': '2026-09-15'},
                {'op': 'set', 'slot': 'time', 'value': '10:00'}])
            voice.task.consume({'intent': 'ambiguous', 'ops': [], 'unclear': False,
                'is_affirmation': False, 'is_negation': False, 'proposed_ops': [],
                'clarification': {'kind': 'ambiguous_value', 'slot': 'doctor', 'item_ids': [], 'coupled_slots': ['time']}})
            voice.pending_request = {'id': 'request_1', 'text': 'earlier dependent alternatives'}
            try:
                await voice.start()
                await voice.playback_task
                before = deepcopy((voice.task.__dict__, voice.pending_request))
                initial_renders = list(renders)
                await voice.queue.put(Segment(bytes([1, 0]) * 512, 0, 320, 1728, 320))
                await asyncio.wait_for(voice.queue.join(), 2)
                assert voice.status == 'error' and voice.done.is_set()
                assert (voice.task.__dict__, voice.pending_request) == before
                assert renders == initial_renders and not voice.task.confirmed
                assert len(provider.requests) == 3 and len(stt.calls) == 1
                assert any(event['type'] == 'error' for event in events)
                assert not any(event['type'] == 'warning' for event in events)
                assert_safe(events)
                assert_safe(voice.turns)
            finally:
                await voice.close()
            report, = tmp_path.rglob('demo_session_*.json')
            assert_safe(report.read_text(encoding='utf-8'))
    asyncio.run(run())
