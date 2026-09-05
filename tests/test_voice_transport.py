"""Offline protocol playback; no server, model or audio device is contacted."""

import asyncio
import io
import json
from types import SimpleNamespace
import wave

import pytest
import websockets
from websockets.exceptions import ConnectionClosedOK
from websockets.frames import Close

from bench.voice_transport import error_messages, run_transport, verified_wav


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(bytes(32))
    return output.getvalue()


def clean_close():
    return ConnectionClosedOK(Close(1000, ''), Close(1000, ''), True)


class ProtocolSocket:
    def __init__(self, mode='complete'):
        self.queue = asyncio.Queue()
        self.mode = mode
        self.closed = False
        self.greeting_complete = False
        self.pcm_frames = 0
        self.frames_after_stop = 0
        self.sender_stopped_before_stop = False
        self.controls = []
        self.raw = wav_bytes()
        self.emit(type='ready')

    def emit(self, **event):
        self.queue.put_nowait(json.dumps(event))

    def audio(self, playback_id, action):
        self.emit(type='assistant_text', action=action, text='Offline test reply')
        self.emit(type='audio_start', playback_id=playback_id, action=action, bytes=len(self.raw))
        # Deliberately split the header and body, to verify complete aggregation.
        for part in (self.raw[:7], self.raw[7:43], self.raw[43:]):
            self.queue.put_nowait(part)
        self.emit(type='audio_end', playback_id=playback_id)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self.queue.get()
        if message is None:
            raise StopAsyncIteration
        return message

    async def send(self, payload):
        if isinstance(payload, bytes):
            if self.closed:
                self.frames_after_stop += 1
                raise clean_close()
            if self.mode == 'early_send_close':
                raise clean_close()
            if self.greeting_complete:
                self.pcm_frames += 1
                if self.pcm_frames == 3:
                    self.emit(type='transcript', text='Offline fixture')
                    self.audio('reply', 'ask')
            return
        event = json.loads(payload)
        self.controls.append(event)
        if event['type'] == 'start':
            self.emit(type='started', session_id='offline-session')
            if self.mode == 'early_receive_close':
                self.queue.put_nowait(None)
                return
            self.audio('greeting', 'greeting')
        elif event['type'] == 'playback_finished':
            if event['playback_id'] == 'greeting':
                self.greeting_complete = True
            self.emit(type='state', state='LISTENING')
        elif event['type'] == 'stop':
            # A task still sleeping when stop closes the socket caused the
            # original false failure. It must now be cancelled and joined first.
            self.sender_stopped_before_stop = not any(
                task.get_name() == 'voice_transport_input' and not task.done()
                for task in asyncio.all_tasks()
            )
            self.closed = True
            self.emit(type='done', status='interrupted', slots={})
            self.queue.put_nowait(None)


def execute(monkeypatch, tmp_path, mode='complete'):
    socket = ProtocolSocket(mode)
    monkeypatch.setattr(websockets, 'connect', lambda *args, **kwargs: socket)
    report = {'timings_ms': {}, 'events': [], 'outputs': [], 'frames_sent': 0}
    config = {'runtime': {'frame_ms': 32, 'sample_rate_hz': 16000},
              'endpointing': {'base_silence_ms': 0, 'max_wait_ms': 0}}
    args = SimpleNamespace(domain='clinic', timeout_seconds=2)
    return socket, report, run_transport(args, bytes(1024), config, report, tmp_path)


def test_completed_reply_stops_sender_before_clean_session_close(monkeypatch, tmp_path):
    socket, report, run = execute(monkeypatch, tmp_path)
    asyncio.run(run)
    assert socket.sender_stopped_before_stop
    assert socket.frames_after_stop == 0
    assert socket.pcm_frames == 3
    assert report['server_status'] == 'interrupted'
    assert 'stop_sent' in report['timings_ms']
    assert 'server_done' in report['timings_ms']
    assert [event['type'] for event in socket.controls] == [
        'start', 'playback_finished', 'playback_finished', 'stop',
    ]
    assert len(report['outputs']) == 2
    for output in report['outputs']:
        assert output['received_bytes'] == output['announced_bytes'] == len(socket.raw)
        assert output['binary_frames'] == 3
        assert output['status'] == 'simulated_playback_acknowledged'
        assert output['device_playback_tested'] is False
        assert 'server_ack_processed_ms' in output
        assert verified_wav(socket.raw)[1]['samples'] == output['samples']


@pytest.mark.parametrize('mode,expected', [
    ('early_send_close', 'ConnectionClosedOK'),
    ('early_receive_close', 'WebSocket closed without a final session result'),
])
def test_clean_close_before_completed_lifecycle_remains_failure(monkeypatch, tmp_path, mode, expected):
    socket, report, run = execute(monkeypatch, tmp_path, mode)
    with pytest.raises(ExceptionGroup) as failure:
        asyncio.run(run)
    assert any(expected in message for message in error_messages(failure.value))
    assert 'stop_sent' not in report['timings_ms']
    assert not any(event['type'] == 'stop' for event in socket.controls)
