"""Actual server connection/session lifecycle with offline adapters and socket."""

import asyncio
from contextlib import AsyncExitStack
import json

import pytest
from starlette.websockets import WebSocketDisconnect

from engine.config import ROOT, load_config
from engine.endpointing import Segment
from engine.pipeline import VoiceSession
from engine.responder import RenderedAudio
from engine.stt import Transcript, wav_bytes
import engine.server as server


class Socket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.listening = asyncio.Event()
        self.disconnected = False
        self.termination_event_index = None

    async def receive_json(self):
        return {'type': 'start'}

    async def receive(self):
        message = await self.incoming.get()
        if message['type'] == 'websocket.disconnect':
            self.disconnected = True
            self.termination_event_index = len(self.sent)
        elif message.get('text') == json.dumps({'type': 'stop'}):
            self.termination_event_index = len(self.sent)
        return message

    async def send_json(self, event):
        if self.disconnected:
            raise WebSocketDisconnect()
        self.sent.append(dict(event))
        if event['type'] == 'audio_end':
            self.incoming.put_nowait({'type': 'websocket.receive', 'text': json.dumps({
                'type': 'playback_finished', 'playback_id': event['playback_id'],
            })})
        if event['type'] == 'state' and event['state'] == 'LISTENING':
            self.listening.set()

    async def send_bytes(self, data):
        if self.disconnected:
            raise WebSocketDisconnect()

    async def close(self, **kwargs):
        pass

    def terminate(self, mode):
        self.incoming.put_nowait({'type': 'websocket.disconnect'} if mode == 'disconnect' else
            {'type': 'websocket.receive', 'text': json.dumps({'type': 'stop'})})


def setup(monkeypatch, tmp_path, domain, *, blocked=None, constructor_failure=None, cleanup_failure=None):
    selected = load_config(ROOT / 'configs' / f'{domain}.json')
    created, closed, voices = [], [], []
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def wait_if_blocked(stage):
        if stage == blocked:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    class Resource:
        def __init__(self, kind):
            if constructor_failure == kind:
                raise RuntimeError('Offline constructor failure')
            self.kind = kind
            self.calls = []
            created.append(kind)

        async def close(self):
            closed.append(self.kind)
            if cleanup_failure == self.kind:
                raise RuntimeError('Offline cleanup failure')

    class ASR(Resource):
        def __init__(self, *args, **kwargs):
            super().__init__('stt')

        async def transcribe(self, pcm):
            await wait_if_blocked('transcribing')
            return Transcript('offline request', .99, 'fixture')

    class Router(Resource):
        def __init__(self, *args, **kwargs):
            super().__init__('router')

        async def route(self, text, state):
            await wait_if_blocked('routing')
            return {'intent': 'task', 'ops': [], 'unclear': False,
                    'is_affirmation': False, 'is_negation': False}

    class Bank(Resource):
        def __init__(self, config, *args, **kwargs):
            super().__init__('bank')
            self.config = config

        async def render(self, action, values):
            if action.kind != 'greeting':
                await wait_if_blocked('synthesizing')
            return RenderedAudio(wav_bytes(bytes(1024), self.config['runtime']), 512, 512,
                                 ['offline-fixture'], 'Offline fixture reply')

    def voice_factory(*args, **kwargs):
        voice = VoiceSession(*args, **kwargs)
        voices.append(voice)
        return voice

    monkeypatch.setattr(server, 'ROOT', tmp_path)
    monkeypatch.setattr(server, 'demo_issues', lambda config: [])
    monkeypatch.setattr(server, 'SpeechToText', ASR)
    monkeypatch.setattr(server, 'Router', Router)
    monkeypatch.setattr(server, 'DemoVoice', Bank)
    monkeypatch.setattr(server, 'SileroVAD', lambda config: lambda pcm: 0.0)
    monkeypatch.setattr(server, 'VoiceSession', voice_factory)
    return selected, created, closed, voices, entered, cancelled


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
@pytest.mark.parametrize('termination', ['stop', 'disconnect'])
@pytest.mark.parametrize('stage', ['transcribing', 'routing', 'synthesizing'])
def test_connection_stop_cancels_blocked_work_and_closes_resources(monkeypatch, tmp_path, domain, termination, stage):
    async def run():
        selected, created, closed, voices, entered, cancelled = setup(monkeypatch, tmp_path, domain, blocked=stage)
        socket = Socket()
        async with asyncio.timeout(3):
            connection = asyncio.create_task(server.voice_connection(socket, selected, demo=True))
            await socket.listening.wait()
            voice, = voices
            await voice.queue.put(Segment(bytes([1, 0]) * 512, 0, 320, 1728, 320))
            await entered.wait()
            socket.terminate(termination)
            await connection
        assert cancelled.is_set()
        assert created == ['stt', 'router', 'bank']
        assert sorted(closed) == sorted(created)
        assert voice.closed and voice.worker.done()
        assert not voice.task.confirmed
        assert socket.termination_event_index is not None
        assert not any(event['type'] in {'progress', 'assistant_text', 'audio_start', 'audio_end'}
                       for event in socket.sent[socket.termination_event_index:])
        assert not any(event.get('status') in {'completed', 'demo_completed'} for event in socket.sent)
        saved, = tmp_path.rglob('demo_session_*.json')
        result = json.loads(saved.read_text(encoding='utf-8'))
        assert result['status'] == 'interrupted' and not result['confirmed']
        assert result['session_id'] == voice.session_id

    asyncio.run(run())


@pytest.mark.parametrize('failed_constructor,expected', [('router', ['stt']), ('bank', ['stt', 'router'])])
def test_constructor_failure_closes_previously_created_resources(monkeypatch, tmp_path, failed_constructor, expected):
    async def run():
        selected, created, closed, voices, _, _ = setup(monkeypatch, tmp_path, 'pizza', constructor_failure=failed_constructor)
        socket = Socket()
        async with asyncio.timeout(3):
            try:
                await server.voice_connection(socket, selected, demo=True)
            except RuntimeError:
                pass  # Cleanup must hold whether setup failure is returned or propagated.
        assert created == expected
        assert sorted(closed) == sorted(expected), 'An initialization failure must close every earlier adapter.'
        assert voices == []
        assert not list(tmp_path.rglob('demo_session_*.json'))

    asyncio.run(run())


@pytest.mark.parametrize('failure', ['stt', 'bank', 'voice_save'])
def test_one_cleanup_failure_does_not_skip_other_resources(monkeypatch, tmp_path, failure):
    async def run():
        selected, created, closed, voices, _, _ = setup(monkeypatch, tmp_path, 'clinic', cleanup_failure=failure)
        socket = Socket()
        async with asyncio.timeout(3):
            connection = asyncio.create_task(server.voice_connection(socket, selected, demo=True))
            await socket.listening.wait()
            if failure == 'voice_save':
                def fail_save():
                    raise OSError('Offline fixture save failure')
                monkeypatch.setattr(voices[0], 'save', fail_save)
            socket.terminate('disconnect')
            try:
                await connection
            except (RuntimeError, OSError):
                pass
        assert created == ['stt', 'router', 'bank']
        assert sorted(closed) == sorted(created), 'Later adapters still need cleanup after the first close raises.'
        assert voices[0].closed
        saved = list(tmp_path.rglob('demo_session_*.json'))
        if failure == 'voice_save':
            assert not saved
        else:
            assert len(saved) == 1
            assert json.loads(saved[0].read_text(encoding='utf-8'))['status'] == 'interrupted'

    asyncio.run(run())


@pytest.mark.parametrize('cancellation_count', [1, 2])
@pytest.mark.parametrize('delayed_closer_fails', [False, True])
def test_cleanup_finishes_all_closers_before_propagating_cancellation(cancellation_count, delayed_closer_fails):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        callbacks = []
        stack = AsyncExitStack()

        async def later(name):
            callbacks.append(name)

        async def delayed():
            callbacks.append('bank_started')
            entered.set()
            await release.wait()
            callbacks.append('bank_finished')
            if delayed_closer_fails:
                raise RuntimeError('Offline fixture closer failure')

        stack.push_async_callback(later, 'stt')
        stack.push_async_callback(later, 'router')
        stack.push_async_callback(delayed)
        async with asyncio.timeout(3):
            caller = asyncio.create_task(server.finish_voice_cleanup(stack))
            await entered.wait()
            for _ in range(cancellation_count):
                assert caller.cancel()
                await asyncio.sleep(0)
                assert not caller.done(), 'Cancellation must wait for registered resource cleanup.'
                assert callbacks == ['bank_started']
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert caller.cancelled()
            assert callbacks == ['bank_started', 'bank_finished', 'router', 'stt']
            await stack.aclose()
            assert callbacks == ['bank_started', 'bank_finished', 'router', 'stt']

    asyncio.run(run())
