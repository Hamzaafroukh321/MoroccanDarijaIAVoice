"""Recovery handshake and fresh playback authority through the real server."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config, reply_text
from engine.endpointing import Segment
from engine.pipeline import VoiceSession
from engine.recovery import RecoveryStore
from engine.responder import RenderedAudio
from engine.router import Router
from engine.scoped_task import ScopedTaskState
from engine.stt import Transcript, wav_bytes
import engine.server as server


class Socket:
    def __init__(self, start=None, auto_ack=True):
        self.start = start or {'type': 'start'}
        self.auto_ack = auto_ack
        self.incoming = asyncio.Queue()
        self.events = []
        self.playback_end = asyncio.Event()
        self.listening = asyncio.Event()
        self.close_index = None
    async def receive_json(self): return self.start
    async def receive(self): return await self.incoming.get()
    def command(self, **payload):
        self.incoming.put_nowait({'type': 'websocket.receive', 'text': json.dumps(payload)})
    async def send_json(self, event):
        self.events.append(deepcopy(event))
        if event['type'] == 'audio_end':
            self.playback_end.set()
            if self.auto_ack:
                self.command(type='playback_finished', playback_id=event['playback_id'])
        if event['type'] == 'state' and event.get('state') == 'LISTENING':
            self.listening.set()
    async def send_bytes(self, data): pass
    async def close(self, **kwargs): self.close_index = len(self.events)


def setup(monkeypatch, tmp_path, domain):
    selected = load_config(ROOT / f'configs/{domain}.json')
    config = demo_config(selected)
    created, closed, voices = [], [], []
    class ASR:
        def __init__(self, *args, **kwargs): created.append('stt')
        async def transcribe(self, pcm): return Transcript('offline request', .99, 'fixture')
        async def close(self): closed.append('stt')
    class RateLimitedRouter(Router):
        def __init__(self, config, root):
            created.append('router')
            self.mock_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
                httpx.Response(429, text='private provider error')))
            super().__init__(config, root, api_key='offline-fixture', client=self.mock_client)
            self.limiter.reserve = lambda: None
        async def close(self):
            closed.append('router')
            await self.mock_client.aclose()
    class Bank:
        def __init__(self, config, root):
            created.append('bank')
            self.config = config
        async def render(self, action, values):
            return RenderedAudio(wav_bytes(bytes(2048), self.config['runtime']), 1024, 1024,
                ['offline'], reply_text(action, values, self.config['demo']))
        async def close(self): closed.append('bank')
    def factory(*args, **kwargs):
        voice = VoiceSession(*args, **kwargs)
        voices.append(voice)
        return voice
    monkeypatch.setattr(server, 'ROOT', tmp_path)
    monkeypatch.setattr(server, 'demo_issues', lambda config: [])
    monkeypatch.setattr(server, 'SpeechToText', ASR)
    monkeypatch.setattr(server, 'Router', RateLimitedRouter)
    monkeypatch.setattr(server, 'DemoVoice', Bank)
    monkeypatch.setattr(server, 'SileroVAD', lambda config: lambda pcm: 0.0)
    monkeypatch.setattr(server, 'VoiceSession', factory)
    monkeypatch.setattr(server, 'recovery_store', RecoveryStore())
    return selected, config, created, closed, voices


def populate(task, domain, complete):
    if domain == 'clinic':
        task.apply([{'op': 'set', 'slot': 'doctor', 'value': 'doctor_a'},
                    {'op': 'set', 'slot': 'date', 'value': '2026-09-15'}] +
                   ([{'op': 'set', 'slot': 'time', 'value': '11:30'}] if complete else []))
    else:
        task.apply([{'op': 'create', 'item_id': 1, 'slot': None, 'value': None},
            {'op': 'set', 'item_id': 1, 'slot': 'quantity', 'value': 1},
            {'op': 'set', 'item_id': 1, 'slot': 'size', 'value': 'large'},
            {'op': 'set', 'item_id': 1, 'slot': 'toppings', 'value': []}] +
            ([{'op': 'set', 'item_id': None, 'slot': 'drink', 'value': ['water']}] if complete else []))


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
@pytest.mark.parametrize('complete', [False, True])
def test_rate_limit_offers_recovery_before_close_and_resume_has_fresh_playback_authority(monkeypatch, tmp_path, domain, complete):
    async def run():
        selected, config, created, closed, voices = setup(monkeypatch, tmp_path, domain)
        first = Socket()
        async with asyncio.timeout(4):
            connection = asyncio.create_task(server.voice_connection(first, selected, demo=True))
            await first.listening.wait()
            original = voices[0]
            populate(original.task, domain, complete)
            committed = deepcopy(original.task.values)
            original.pending_request = {'id': 'request_1', 'text': 'discard this retained original'}
            old_playback_id = next(event['playback_id'] for event in first.events if event['type'] == 'audio_end')
            await original.queue.put(Segment(bytes([1, 0]) * 512, 0, 320, 1728, 320))
            await connection
            error = next(event for event in first.events if event['type'] == 'error')
            offered = error['recovery']
            assert first.events.index(error) < first.close_index
            assert 'rate limit' in error['message'] and 'private provider' not in error['message']
            assert offered['slots'] == committed and offered['discarded_pending'] is True
            assert original.closed and original.recovery_frozen
            assert sorted(created) == sorted(closed)

            second = Socket({'type': 'start', 'resume_token': offered['token']}, auto_ack=False)
            resumed_connection = asyncio.create_task(server.voice_connection(second, selected, demo=True))
            await second.playback_end.wait()
            resumed = voices[1]
            assert resumed.task.values == committed and resumed.pending_request is None
            assert resumed.task.pending_proposal is None and resumed.task.pending_clarification is None
            started = next(event for event in second.events if event['type'] == 'started')
            assert started['resumed'] and started['discarded_pending']
            assert resumed.session_id != original.session_id
            audio = next(event for event in second.events if event['type'] == 'audio_start')
            assert audio['action'] == ('readback' if complete else 'ask')
            assert resumed.task.readback_version is None and not resumed.task.confirmed
            await resumed.playback_finished(old_playback_id)
            assert not resumed.playback_done.is_set()
            assert resumed.task.readback_version is None and not resumed.task.confirmed
            second.command(type='playback_finished', playback_id=audio['playback_id'])
            await resumed.playback_task
            assert (resumed.task.readback_version == resumed.task.version) is complete
            assert not resumed.task.confirmed
            second.command(type='stop')
            await resumed_connection
            assert sorted(created) == sorted(closed)
            with pytest.raises(ValueError):
                server.recovery_store.restore(offered['token'], config)
        reports = [path.read_text(encoding='utf-8') for path in tmp_path.rglob('demo_session_*.json')]
        assert len(reports) == 2
        assert all(offered['token'] not in text for text in reports)
        latest = next(json.loads(text) for text in reports if json.loads(text)['session_id'] == resumed.session_id)
        assert latest['resumed_from'] == original.session_id
        assert not latest['confirmed']
    asyncio.run(run())


@pytest.mark.parametrize('token', ['invalid', None])
def test_invalid_resume_token_rejects_before_provider_allocation(monkeypatch, tmp_path, token):
    async def run():
        selected, _, created, closed, voices = setup(monkeypatch, tmp_path, 'clinic')
        socket = Socket({'type': 'start', 'resume_token': token})
        await server.voice_connection(socket, selected, demo=True)
        assert created == closed == voices == []
        assert [event['type'] for event in socket.events] == ['ready', 'error']
        assert not socket.events[-1].get('resume_available', False)
        assert socket.close_index is not None
        assert not list(tmp_path.rglob('demo_session_*.json'))
    asyncio.run(run())


@pytest.mark.parametrize('failed_constructor', ['Router', 'VoiceSession'])
def test_failed_resume_resource_setup_preserves_token_and_closes_prior_resources(monkeypatch, tmp_path, failed_constructor):
    async def run():
        selected, config, created, closed, voices = setup(monkeypatch, tmp_path, 'clinic')
        state = ScopedTaskState(config)
        populate(state, 'clinic', True)
        offered = server.recovery_store.offer(config, state, 'old-session')
        def fail(*args, **kwargs):
            raise RuntimeError('offline constructor failure')
        monkeypatch.setattr(server, failed_constructor, fail)
        socket = Socket({'type': 'start', 'resume_token': offered['token']})
        await server.voice_connection(socket, selected, demo=True)
        assert created == (['stt'] if failed_constructor == 'Router' else ['stt', 'router', 'bank'])
        assert sorted(closed) == sorted(created)
        assert not voices
        error = next(event for event in socket.events if event['type'] == 'error')
        assert error.get('resume_available') is True
        fresh = server.recovery_store.restore(offered['token'], config).task
        assert fresh.values == state.values and not fresh.confirmed
    asyncio.run(run())


@pytest.mark.parametrize('invalid_reason', ['expired_before_start', 'expired_during_setup', 'evicted_during_setup', 'reused'])
def test_unusable_token_never_advertises_resume_availability(monkeypatch, tmp_path, invalid_reason):
    async def run():
        selected, config, created, closed, _ = setup(monkeypatch, tmp_path, 'clinic')
        now = [100.0]
        registry = RecoveryStore(clock=lambda: now[0])
        monkeypatch.setattr(server, 'recovery_store', registry)
        state = ScopedTaskState(config)
        populate(state, 'clinic', True)
        offered = registry.offer(config, state, 'old-session')
        if invalid_reason == 'expired_before_start':
            now[0] += 1800
        elif invalid_reason == 'reused':
            registry.restore(offered['token'], config)
        else:
            def expire_and_fail(*args, **kwargs):
                if invalid_reason == 'evicted_during_setup':
                    for index in range(32):
                        registry.offer(config, state, f'other-session-{index}')
                else:
                    now[0] += 1800
                raise RuntimeError('offline setup failed after token expiry')
            monkeypatch.setattr(server, 'Router', expire_and_fail)
        socket = Socket({'type': 'start', 'resume_token': offered['token']})
        await server.voice_connection(socket, selected, demo=True)
        error = next(event for event in socket.events if event['type'] == 'error')
        assert not error.get('resume_available', False)
        assert 'recovery' not in error
        assert created == closed == (['stt'] if invalid_reason.endswith('_during_setup') else [])
        with pytest.raises(ValueError):
            registry.restore(offered['token'], config)
    asyncio.run(run())
