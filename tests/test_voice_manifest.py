"""Manifest and multi-turn transport controls with local fake audio/socket only."""

import asyncio
import io
import json
import sys
from types import SimpleNamespace
import wave

import pytest
import websockets

from bench import voice_manifest
from bench import voice_transport


def wav_bytes(samples=512, marker=1):
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(bytes([marker, 0]) * samples)
    return output.getvalue()


def manifest_file(tmp_path, *, count=3, samples=512):
    turns = []
    for index in range(count):
        path = tmp_path / f'input_{index}.wav'
        path.write_bytes(wav_bytes(samples, index + 1))
        turns.append(dict(audio=path.name, source_kind='synthetic_diagnostic',
                          expected_action='ask', expected_slots={'step': index + 1}))
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'domain': 'clinic', 'turns': turns}), encoding='utf-8')
    return path


def test_manifest_resolves_relative_wavs_and_retains_provenance(tmp_path):
    path = manifest_file(tmp_path)
    loaded = voice_manifest.load_manifest(path)
    assert loaded['domain'] == 'clinic' and len(loaded['turns']) == 3
    for index, turn in enumerate(loaded['turns']):
        assert turn['source'] == str((tmp_path / f'input_{index}.wav').resolve())
        assert turn['source_kind'] == 'synthetic_diagnostic'
        assert len(turn['pcm']) == 1024 and len(turn['source_sha256']) == 64
        assert turn['input_audio']['duration_seconds'] == .032


@pytest.mark.parametrize('case', ['empty', 'too_many', 'source_kind_missing', 'source_kind_invalid',
    'too_long_one', 'too_long_total', 'corrupt_audio', 'missing_audio', 'timeout'])
def test_invalid_manifest_rejected_without_network(monkeypatch, tmp_path, case):
    monkeypatch.setattr(websockets, 'connect', lambda *a, **k: pytest.fail('Unexpected network allocation'))
    count = 0 if case == 'empty' else 6 if case == 'too_many' else 4 if case == 'too_long_total' else 1
    samples = 320001 if case == 'too_long_one' else 256000 if case == 'too_long_total' else 512
    path = manifest_file(tmp_path, count=count, samples=samples)
    data = json.loads(path.read_text())
    if case == 'source_kind_missing':
        del data['turns'][0]['source_kind']
    elif case == 'source_kind_invalid':
        data['turns'][0]['source_kind'] = 'human_reviewed'
    elif case == 'corrupt_audio':
        (tmp_path / 'input_0.wav').write_bytes(b'not a WAV')
    elif case == 'missing_audio':
        data['turns'][0]['audio'] = 'missing.wav'
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises((voice_manifest.ManifestError, voice_transport.TransportError, OSError)):
        voice_manifest.load_manifest(path, timeout_seconds=181 if case == 'timeout' else 180)


class FastPacer:
    def __init__(self, *args, **kwargs):
        self.resets = 0
        self.max_lag_seconds = 0

    async def wait_next(self, stopped=lambda: False):
        await asyncio.sleep(0)
        return not stopped()


class MultiTurnSocket:
    def __init__(self, mode='complete'):
        self.mode = mode
        self.queue = asyncio.Queue()
        self.controls = []
        self.source_markers = []
        self.tail_frames = {}
        self.reply_acks = []
        self.greeting_acked = False
        self.current = None
        self.replied = set()
        self.raw = wav_bytes(samples=16)
        self.emit(type='ready')

    def emit(self, **event):
        self.queue.put_nowait(json.dumps(event))

    def audio(self, playback_id, action):
        self.emit(type='assistant_text', playback_id=playback_id, action=action, text='Offline fixture')
        self.emit(type='audio_start', playback_id=playback_id, action=action, bytes=len(self.raw))
        for part in (self.raw[:9], self.raw[9:45], self.raw[45:]):
            self.queue.put_nowait(part)
        self.emit(type='audio_end', playback_id=playback_id)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self.queue.get()
        if message is None:
            raise StopAsyncIteration
        return message

    async def send(self, payload):
        if isinstance(payload, bytes):
            assert len(payload) == 1024
            if any(payload):
                assert self.greeting_acked, 'Source sent before greeting ACK'
                marker = payload[0]
                assert marker not in self.source_markers, 'Source repeated unexpectedly'
                if self.source_markers:
                    assert self.source_markers[-1] in self.reply_acks, 'Next source sent before reply ACK'
                self.source_markers.append(marker)
                self.current = marker
                self.tail_frames[marker] = 0
            elif self.current is not None and self.current not in self.replied:
                self.tail_frames[self.current] += 1
                if self.tail_frames[self.current] == 2:
                    self.replied.add(self.current)
                    self.emit(type='endpoint', speech_end_ms=32 * self.current)
                    if self.mode == 'multiple_endpoints':
                        self.emit(type='endpoint', speech_end_ms=64 * self.current)
                    self.emit(type='transcript', text='Offline fixture',
                              provider='whisper' if self.mode == 'wrong_provider' else 'moulsot')
                    action = ('accepted' if self.mode in {'final_accepted', 'final_accepted_ack_yields'} and self.current == 3 else
                              'readback' if self.mode == 'wrong_action' else 'ask')
                    self.audio(f'reply-{self.current}', action)
            return
        event = json.loads(payload)
        self.controls.append(event)
        if event['type'] == 'start':
            self.emit(type='started', session_id='offline-multiturn')
            self.audio('greeting', 'greeting')
        elif event['type'] == 'playback_finished':
            playback_id = event['playback_id']
            step = 0 if playback_id == 'greeting' else int(playback_id.split('-')[1])
            if step:
                self.reply_acks.append(step)
            else:
                self.greeting_acked = True
            slots = {'step': -1 if self.mode == 'wrong_state' and step else step}
            if self.mode == 'wrong_state_type' and step:
                slots = {'step': True}
            self.emit(type='state', state='LISTENING', slots=slots)
            completion = dict(type='playback_complete', playback_id=playback_id,
                action=('greeting' if not step else 'accepted' if self.mode in {'final_accepted', 'final_accepted_ack_yields'} and step == 3 else
                        'readback' if self.mode == 'wrong_action' else 'ask'),
                slots=slots, pending_clarification=None, pending_proposal=None,
                version=step, readback_complete=False)
            if self.mode == 'completion_action_mismatch' and step:
                completion['action'] = 'accepted'
            self.emit(**completion)
            # Repeated state and completion events belong to the same WAV;
            # they must not complete or consume the next manifest turn.
            self.emit(type='state', state='LISTENING', slots=slots)
            self.emit(**completion)
            if self.mode == 'final_accepted_ack_yields' and step == 3:
                self.emit(type='done', status='demo_completed', slots=slots)
                self.queue.put_nowait(None)
            if self.mode in {'ack_send_yields', 'ack_send_fails', 'final_accepted_ack_yields'}:
                # A real send can yield after the bytes are delivered. Let the
                # receiver observe completion before send() returns locally.
                await asyncio.sleep(0)
                if self.mode == 'ack_send_fails' and step:
                    raise RuntimeError('Offline ACK send failed after queued completion')
            if self.mode == 'final_accepted' and step == 3:
                def finish_naturally():
                    self.emit(type='done', status='demo_completed', slots=slots)
                    self.queue.put_nowait(None)
                # Leave a real scheduling gap after the final completion. An
                # eager stop would incorrectly replace this with interrupted.
                asyncio.get_running_loop().call_later(.01, finish_naturally)
            if self.mode == 'early_done' and step == 1:
                self.emit(type='done', status='handoff', slots=slots)
                self.queue.put_nowait(None)
        elif event['type'] == 'stop':
            self.emit(type='done', status='interrupted', slots={'step': self.current})
            self.queue.put_nowait(None)


def execute(monkeypatch, tmp_path, mode='complete', *, trailing_ms=0):
    manifest = voice_manifest.load_manifest(manifest_file(tmp_path))
    if mode in {'final_accepted', 'final_accepted_ack_yields'}:
        manifest['turns'][-1]['expected_action'] = 'accepted'
    socket = MultiTurnSocket(mode)
    monkeypatch.setattr(websockets, 'connect', lambda *args, **kwargs: socket)
    monkeypatch.setattr(voice_transport, 'FramePacer', FastPacer)
    config = {'runtime': {'frame_ms': 32, 'sample_rate_hz': 16000},
              'endpointing': {'base_silence_ms': 0, 'max_wait_ms': trailing_ms}}
    args = SimpleNamespace(domain='clinic', timeout_seconds=2)
    report = {'timings_ms': {}, 'events': [], 'outputs': [], 'frames_sent': 0}
    return socket, report, voice_transport.run_manifest_transport(
        args, manifest['turns'], config, report, tmp_path)


@pytest.mark.parametrize('mode', ['complete', 'ack_send_yields'])
def test_multi_turns_wait_for_each_verified_reply_without_duplicate_state_skip(monkeypatch, tmp_path, mode):
    socket, report, run = execute(monkeypatch, tmp_path, mode)
    asyncio.run(run)
    assert socket.source_markers == [1, 2, 3]
    assert socket.reply_acks == [1, 2, 3]
    assert all(value == 2 for value in socket.tail_frames.values())
    assert [turn['status'] for turn in report['turns']] == ['completed'] * 3
    assert len(report['outputs']) == 4
    assert [output['turn_index'] for output in report['outputs']] == [None, 0, 1, 2]
    for output in report['outputs']:
        assert output['received_bytes'] == output['announced_bytes'] == len(socket.raw)
        assert output['binary_frames'] == 3
        assert output['device_playback_tested'] is False
        assert 'server_ack_processed_ms' in output
    assert socket.controls[-1]['type'] == 'stop'


@pytest.mark.parametrize('mode', ['early_done', 'wrong_action', 'wrong_state', 'wrong_state_type',
                                  'multiple_endpoints', 'wrong_provider', 'completion_action_mismatch'])
def test_multi_turn_protocol_failure_cannot_continue_to_next_source(monkeypatch, tmp_path, mode):
    socket, report, run = execute(monkeypatch, tmp_path, mode)
    with pytest.raises(ExceptionGroup) as failed:
        asyncio.run(run)
    messages = ' '.join(voice_transport.error_messages(failed.value)).lower()
    expected = {'early_done': 'finished', 'wrong_action': 'action', 'wrong_state': 'state',
                'wrong_state_type': 'state', 'completion_action_mismatch': 'action',
                'multiple_endpoints': 'endpoint', 'wrong_provider': 'moulsot'}
    assert expected[mode] in messages
    assert socket.source_markers == [1]
    assert not any(event['type'] == 'stop' for event in socket.controls)
    assert not all(turn['status'] == 'completed' for turn in report['turns'])


def test_failed_ack_send_cannot_complete_turn_from_queued_server_event(monkeypatch, tmp_path):
    socket, report, run = execute(monkeypatch, tmp_path, 'ack_send_fails')
    with pytest.raises(ExceptionGroup) as failed:
        asyncio.run(run)
    assert 'ACK send failed' in ' '.join(voice_transport.error_messages(failed.value))
    assert socket.source_markers == [1]
    assert report['turns'][0]['status'] != 'completed'
    assert not any(event['type'] == 'stop' for event in socket.controls)


@pytest.mark.parametrize('mode', ['final_accepted', 'final_accepted_ack_yields'])
def test_final_accepted_reply_waits_for_natural_done_without_sending_stop(monkeypatch, tmp_path, mode):
    socket, report, run = execute(monkeypatch, tmp_path, mode)
    asyncio.run(run)
    assert socket.source_markers == socket.reply_acks == [1, 2, 3]
    assert all(turn['status'] == 'completed' for turn in report['turns'])
    assert report['server_status'] == 'demo_completed'
    assert not any(event['type'] == 'stop' for event in socket.controls)


@pytest.mark.parametrize('mode', ['final_accepted_ack_yields', 'early_done'])
def test_only_final_natural_terminal_can_truncate_conservative_silence_tail(monkeypatch, tmp_path, mode):
    socket, report, run = execute(monkeypatch, tmp_path, mode, trailing_ms=1000)
    class BoundedPacer(FastPacer):
        async def wait_next(self, stopped=lambda: False):
            # Keep this offline test short while leaving enough scheduling time
            # for reply/done to beat the deliberately conservative 34-frame tail.
            await asyncio.sleep(.001)
            return not stopped()
    monkeypatch.setattr(voice_transport, 'FramePacer', BoundedPacer)
    if mode == 'early_done':
        with pytest.raises(ExceptionGroup) as failed:
            asyncio.run(run)
        assert 'finished' in ' '.join(voice_transport.error_messages(failed.value)).lower()
        assert socket.source_markers == [1]
        assert not all(turn['status'] == 'completed' for turn in report['turns'])
        return
    asyncio.run(run)
    final = report['turns'][-1]
    assert final['source_complete'] and final['status'] == 'completed'
    assert final['tail_complete'] is False and final['tail_truncated_by_terminal'] is True
    assert 2 <= final['tail_frames_sent'] < report['trailing_silence_ms'] // 32
    assert final['tail_ms_sent'] == final['tail_frames_sent'] * 32
    assert report['server_status'] == 'demo_completed'
    assert socket.source_markers == socket.reply_acks == [1, 2, 3]
    assert not any(event['type'] == 'stop' for event in socket.controls)


def test_invalid_loaded_source_fails_before_socket_allocation(monkeypatch, tmp_path):
    loaded = voice_manifest.load_manifest(manifest_file(tmp_path))
    loaded['turns'][1]['pcm'] = b''
    monkeypatch.setattr(websockets, 'connect', lambda *a, **k: pytest.fail('Connected before validating all inputs'))
    args = SimpleNamespace(domain='clinic', timeout_seconds=2)
    config = {'runtime': {'frame_ms': 32, 'sample_rate_hz': 16000},
              'endpointing': {'base_silence_ms': 0, 'max_wait_ms': 0}}
    report = {'timings_ms': {}, 'events': [], 'outputs': [], 'frames_sent': 0}
    with pytest.raises((voice_transport.TransportError, voice_manifest.ManifestError)):
        asyncio.run(voice_transport.run_manifest_transport(args, loaded['turns'], config, report, tmp_path))


def test_direct_manifest_timeout_cannot_exceed_configured_session_limit(monkeypatch, tmp_path):
    loaded = voice_manifest.load_manifest(manifest_file(tmp_path))
    monkeypatch.setattr(websockets, 'connect', lambda *a, **k: pytest.fail('Connected before checking session limit'))
    args = SimpleNamespace(domain='clinic', timeout_seconds=2)
    config = {'runtime': {'frame_ms': 32, 'sample_rate_hz': 16000},
              'endpointing': {'base_silence_ms': 0, 'max_wait_ms': 0},
              'engine': {'max_session_ms': 1000}}
    report = {'timings_ms': {}, 'events': [], 'outputs': [], 'frames_sent': 0}
    with pytest.raises((voice_transport.TransportError, voice_manifest.ManifestError), match='(?i)session|timeout'):
        asyncio.run(voice_transport.run_manifest_transport(args, loaded['turns'], config, report, tmp_path))


@pytest.mark.parametrize('flags', [[], ['--execute', '--check']])
def test_manifest_cli_is_offline_by_default_or_explicit_check(monkeypatch, tmp_path, capsys, flags):
    path = manifest_file(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('Validation allocated a transport connection or runner')
    monkeypatch.setattr(websockets, 'connect', forbidden)
    monkeypatch.setattr(voice_transport, 'run_manifest_transport', forbidden)
    monkeypatch.setattr(voice_transport, 'run_transport', forbidden)
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'groq')
    monkeypatch.setattr(sys, 'argv', ['voice_transport.py', '--manifest', str(path), *flags])
    assert voice_transport.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'manifest_validated_only' and report['network_used'] is False
    assert report['evaluation_eligible'] is False and report['execute_required'] is True
    assert report['expectations_kind'] == 'diagnostic_unreviewed'
    assert len(report['turns']) == 3 and all('pcm' not in turn for turn in report['turns'])


@pytest.mark.parametrize('failure', ['later_missing_source', 'domain_mismatch'])
def test_manifest_cli_execute_validates_entire_input_before_transport(monkeypatch, tmp_path, capsys, failure):
    path = manifest_file(tmp_path)
    data = json.loads(path.read_text())
    flags = []
    if failure == 'later_missing_source':
        data['turns'][2]['audio'] = 'missing.wav'
        path.write_text(json.dumps(data), encoding='utf-8')
    else:
        flags = ['--domain', 'pizza']
    def forbidden(*args, **kwargs):
        pytest.fail('Connected before validating all manifest sources and domain')
    monkeypatch.setattr(websockets, 'connect', forbidden)
    monkeypatch.setattr(voice_transport, 'run_manifest_transport', forbidden)
    monkeypatch.setattr(sys, 'argv', ['voice_transport.py', '--manifest', str(path), '--execute', *flags])
    assert voice_transport.main() == 2
    assert 'NOT RUN:' in capsys.readouterr().out
