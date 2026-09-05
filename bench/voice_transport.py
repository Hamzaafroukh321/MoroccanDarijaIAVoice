"""Bounded local voice transport checks with supplied WAVs and simulated ACKs.

This runs the server's configured ASR/router/TTS path, including Silero VAD.
It does not use a microphone, WebAudio, an output device or a human listener.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import time
from uuid import uuid4
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import ROOT, load_config
from engine.demo import demo_config


MAX_AUDIO_BYTES = 16 * 1024 * 1024


class TransportError(RuntimeError):
    pass


class FramePacer:
    """Compensate small wake delays, but reset instead of bursting after stalls."""

    def __init__(self, interval_seconds, *, clock=time.perf_counter, sleeper=asyncio.sleep):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError('Frame interval must be finite and positive.')
        self.interval = interval_seconds
        self.clock = clock
        self.sleeper = sleeper
        self.deadline = None
        self.resets = 0
        self.max_lag_seconds = 0.0

    async def wait_next(self, stopped=lambda: False):
        if stopped():
            return False
        now = self.clock()
        if self.deadline is None:
            self.deadline = now
        # Recheck after early wakes as well as late wakes. Cancellation must
        # propagate, and a stop during sleep must never admit another frame.
        while now < self.deadline:
            await self.sleeper(self.deadline - now)
            if stopped():
                return False
            now = self.clock()
        if stopped():
            return False
        lag = max(0.0, now - self.deadline)
        self.max_lag_seconds = max(self.max_lag_seconds, lag)
        if lag >= self.interval:
            self.resets += 1
            self.deadline = now + self.interval
        else:
            self.deadline += self.interval
        return True


def verified_wav(raw):
    """Verify the complete container, not merely the last binary frame."""
    if len(raw) < 44 or raw[:4] != b'RIFF' or raw[8:12] != b'WAVE':
        raise TransportError('Audio is not a complete RIFF/WAVE file.')
    if struct.unpack_from('<I', raw, 4)[0] != len(raw) - 8:
        raise TransportError('RIFF size does not match the complete audio body.')
    try:
        with wave.open(io.BytesIO(raw), 'rb') as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (16000, 1, 2, 'NONE'):
                raise TransportError('Use complete 16 kHz mono PCM16 WAV audio.')
            samples = wav.getnframes()
            pcm = wav.readframes(samples)
    except (wave.Error, EOFError) as exc:
        raise TransportError('The WAV header or sample data is invalid.') from exc
    if samples <= 0 or len(pcm) != samples * 2:
        raise TransportError('Audio samples are empty or truncated.')
    return pcm, {'sample_rate_hz': 16000, 'channels': 1, 'sample_width_bytes': 2,
                 'samples': samples, 'duration_seconds': samples / 16000}


async def run_transport(args, pcm, config, report, output_dir):
    """Compatibility entry point for the original single-WAV diagnostic."""
    _validate_demo_domain(args.domain, config)
    await _run_transport(args, [{'pcm': pcm}], config, report, output_dir, manifest=False)


def _validate_demo_domain(domain, config):
    from bench.voice_manifest import load_demo_domain
    configured = load_demo_domain(domain)
    if domain != config.get('domain_id', domain):
        raise TransportError('Replay domain does not match the configured demo.')
    return configured


async def run_manifest_transport(args, turns, config, report, output_dir):
    from bench.voice_manifest import validate_turns
    validate_turns(turns, args.timeout_seconds)
    configured = _validate_demo_domain(args.domain, config)
    session_limit = min(configured['engine']['max_session_ms'],
                        config.get('engine', {}).get('max_session_ms', 180000))
    if args.timeout_seconds > session_limit / 1000:
        raise TransportError('Manifest timeout must not exceed the configured demo session limit.')
    await _run_transport(args, turns, config, report, output_dir, manifest=True)


async def _run_transport(args, turns, config, report, output_dir, *, manifest):
    # Import only when actually running; --check remains entirely local.
    try:
        import websockets
        from websockets.exceptions import ConnectionClosedOK
    except ImportError as exc:
        raise TransportError('The installed Python environment does not have websockets.') from exc

    frame_ms = config['runtime']['frame_ms']
    frame_bytes = config['runtime']['sample_rate_hz'] * frame_ms // 1000 * 2
    if frame_ms != 32 or frame_bytes != 1024:
        raise TransportError('This diagnostic requires the configured 32 ms / 1024-byte PCM frames.')
    trailing_frames = math.ceil(max(config['endpointing']['base_silence_ms'],
                                   config['endpointing']['max_wait_ms']) / frame_ms) + 2
    report['trailing_silence_ms'] = trailing_frames * frame_ms
    report['websocket_url'] = f'ws://127.0.0.1:8000/ws?domain={args.domain}&mode=demo'
    report['origin'] = 'http://127.0.0.1:8000'
    report['websockets_version'] = websockets.__version__
    started_at = time.perf_counter()
    report['timing_clock'] = 'time.perf_counter'
    report['manifest_mode'] = manifest
    report['bounds_kind'] = 'input_and_session_time_only; not a hard provider-request cap'
    report['expectations_kind'] = 'diagnostic_unreviewed; not native accuracy evaluation'
    turn_records = []
    for index, turn in enumerate(turns):
        turn_records.append({key: value for key, value in turn.items() if key != 'pcm'})
        turn_records[-1].update(index=index, status='unplayed', endpoint_count=0, transcript_count=0,
                               source_complete=False, tail_complete=False,
                               tail_frames_sent=0, tail_ms_sent=0, timings_ms={})
    report['turns'] = turn_records
    turn_finished = [asyncio.Event() for _ in turns]
    turn_input_done = [asyncio.Event() for _ in turns]
    active_turn = None
    started = asyncio.Event()
    greeting_done = asyncio.Event()
    reply_done = asyncio.Event()
    server_done = asyncio.Event()
    stopping = asyncio.Event()
    incoming = None
    last_ack_id = None
    acknowledgements = {}
    outputs_by_id = {}

    def elapsed():
        return round((time.perf_counter() - started_at) * 1000, 3)

    def mark(name):
        report['timings_ms'].setdefault(name, elapsed())

    def mark_turn(index, name):
        turn_records[index]['timings_ms'].setdefault(name, elapsed())
        if index == 0:
            mark(name)

    def complete_turn(index):
        record = turn_records[index]
        if (record['status'] == 'completed' or
                not (record['tail_complete'] or record.get('tail_truncated_by_terminal', False)) or
                'reply_playback_id' not in record):
            return
        if manifest:
            if record['endpoint_count'] != 1 or record['transcript_count'] != 1:
                raise TransportError('Each manifest WAV must produce exactly one endpoint and MoulSot transcript.')
            expected = turns[index]
            if 'expected_action' in expected and expected['expected_action'] != record['reply_action']:
                record['status'] = 'expectation_failed'
                raise TransportError(f'Turn {index + 1} reply action did not match its diagnostic expectation.')
            if ('expected_slots' in expected and
                    json.dumps(expected['expected_slots'], sort_keys=True, allow_nan=False) !=
                    json.dumps(record['state_after_reply'], sort_keys=True, allow_nan=False)):
                record['status'] = 'expectation_failed'
                raise TransportError(f'Turn {index + 1} committed state did not match its diagnostic expectation.')
        record['status'] = 'completed'
        mark_turn(index, 'reply_simulated_complete')
        turn_finished[index].set()

    def acknowledge_completion(output, event):
        # A playback ID may release exactly one input turn. Generic state events
        # cannot prove this in manifest mode; require the explicit server event.
        if output.get('server_ack_processed_ms') is not None:
            return
        if output['status'] == 'simulated_ack_sending':
            output['_pending_completion'] = event
            return
        if output['status'] != 'simulated_playback_acknowledged':
            return
        if manifest and event.get('action') != output['action']:
            raise TransportError('Playback completion action did not match its audio output.')
        output['server_ack_processed_ms'] = elapsed()
        if output['action'] == 'greeting':
            mark('greeting_simulated_complete')
            greeting_done.set()
            return
        index = output['turn_index']
        if index is None:
            raise TransportError('Reply completion arrived without an active input turn.')
        record = turn_records[index]
        if 'reply_playback_id' in record:
            raise TransportError('More than one reply completed for a supplied WAV turn.')
        record.update(reply_playback_id=output['playback_id'], reply_action=output['action'],
                      state_after_reply=event.get('slots', {}))
        if manifest:
            record.update(pending_clarification=event.get('pending_clarification'),
                          pending_proposal=event.get('pending_proposal'),
                          readback_complete=event.get('readback_complete'))
        mark('first_reply_simulated_complete')
        reply_done.set()
        complete_turn(index)
        if manifest and output['action'] in {'accepted', 'handoff'} and index < len(turns) - 1:
            raise TransportError('Server finished early with a terminal reply; remaining supplied turns were not played.')

    async with asyncio.timeout(args.timeout_seconds):
        async with websockets.connect(report['websocket_url'], origin=report['origin'],
                                      open_timeout=min(10, args.timeout_seconds), close_timeout=3,
                                      max_size=MAX_AUDIO_BYTES, max_queue=64) as socket:
            mark('connected')

            async def acknowledge(playback_id, output):
                nonlocal last_ack_id
                # Wait the decoded duration to emulate a device clock. This is
                # explicitly NOT evidence that WebAudio or speakers played it.
                await asyncio.sleep(output['duration_seconds'])
                if output['status'] != 'verified':
                    return
                output['status'] = 'simulated_ack_sending'
                last_ack_id = playback_id
                try:
                    await socket.send(json.dumps({'type': 'playback_finished', 'playback_id': playback_id}))
                except BaseException as exc:
                    output.pop('_pending_completion', None)
                    if output['status'] == 'simulated_ack_sending':
                        output['status'] = 'simulated_ack_cancelled' if isinstance(exc, asyncio.CancelledError) else 'simulated_ack_send_failed'
                    raise
                if output['status'] != 'simulated_ack_sending':
                    output.pop('_pending_completion', None)
                    return
                output['status'] = 'simulated_playback_acknowledged'
                output['simulated_ack_ms'] = elapsed()
                pending = output.pop('_pending_completion', None)
                if pending is not None:
                    acknowledge_completion(output, pending)

            async def receive(group):
                nonlocal incoming, last_ack_id
                async for message in socket:
                    if isinstance(message, bytes):
                        if incoming is None:
                            raise TransportError('Binary audio arrived outside audio_start/audio_end.')
                        incoming['chunks'].append(message)
                        incoming['received_bytes'] += len(message)
                        incoming['binary_frames'] += 1
                        if incoming['received_bytes'] > incoming['announced_bytes']:
                            raise TransportError('Response audio exceeded its announced byte count.')
                        continue
                    try:
                        event = json.loads(message)
                    except (json.JSONDecodeError, TypeError) as exc:
                        raise TransportError('Server returned a non-JSON control message.') from exc
                    if not isinstance(event, dict):
                        raise TransportError('Server control messages must be objects.')
                    report['events'].append({'at_ms': elapsed(), **event})
                    kind = event.get('type')
                    if kind == 'ready':
                        await socket.send(json.dumps({'type': 'start'}))
                    elif kind == 'started':
                        report['session_id'] = event.get('session_id')
                        mark('session_started')
                        started.set()
                    elif kind == 'playback_complete' and manifest:
                        output = outputs_by_id.get(event.get('playback_id'))
                        if output is not None:
                            acknowledge_completion(output, event)
                    elif kind == 'state' and not manifest and event.get('state') in {'LISTENING', 'CONFIRMING', 'DONE'}:
                        # Wait until the server has consumed our simulated ACK,
                        # rather than stopping immediately after socket.send().
                        output = outputs_by_id.get(last_ack_id)
                        if output is not None and output['status'] in {'simulated_ack_sending', 'simulated_playback_acknowledged'}:
                            acknowledge_completion(output, event)
                    elif kind == 'endpoint':
                        if manifest and (active_turn is None or turn_records[active_turn]['status'] == 'completed'):
                            raise TransportError('Unexpected endpoint outside the active supplied WAV turn.')
                        if active_turn is not None:
                            record = turn_records[active_turn]
                            record['endpoint_count'] += 1
                            if manifest and (record['endpoint_count'] > 1 or not record['source_complete']):
                                raise TransportError('A supplied WAV ended early or produced multiple endpoints; split it into individual speech turns.')
                            mark_turn(active_turn, 'first_endpoint_received')
                        # This is the server's actual VAD speech boundary, in
                        # its PCM clock; the file end is recorded separately.
                        report['timings_ms']['input_speech_end_audio_clock'] = event.get('speech_end_ms')
                        mark('first_endpoint_received')
                    elif kind == 'transcript':
                        if manifest:
                            if active_turn is None or event.get('provider') != 'moulsot':
                                raise TransportError('Manifest replay requires an actual MoulSot transcript for each turn.')
                            record = turn_records[active_turn]
                            record['transcript_count'] += 1
                            record['transcript_provider'] = event['provider']
                            if record['transcript_count'] > 1:
                                raise TransportError('More than one transcript arrived for a supplied WAV turn.')
                        if active_turn is not None:
                            mark_turn(active_turn, 'first_transcript_received')
                        mark('first_transcript_received')
                    elif kind == 'assistant_text' and event.get('action') != 'greeting':
                        if active_turn is not None:
                            mark_turn(active_turn, 'first_reply_text_received')
                        mark('first_reply_text_received')
                    elif kind == 'audio_start':
                        if incoming is not None:
                            raise TransportError('A new audio_start interrupted an unfinished audio body without audio_stop.')
                        size = event.get('bytes')
                        playback_id = event.get('playback_id')
                        if type(size) is not int or not 44 <= size <= MAX_AUDIO_BYTES or not isinstance(playback_id, str):
                            raise TransportError('Invalid announced response audio metadata.')
                        if playback_id in outputs_by_id:
                            raise TransportError('Duplicate response playback ID.')
                        index = None if event.get('action') == 'greeting' else active_turn
                        if manifest:
                            if event.get('action') == 'greeting' and outputs_by_id:
                                raise TransportError('Unexpected additional greeting output.')
                            if event.get('action') != 'greeting' and (index is None or
                                    turn_records[index]['endpoint_count'] != 1 or
                                    any(output['turn_index'] == index for output in outputs_by_id.values())):
                                raise TransportError('Unexpected extra or unassociated reply audio.')
                        last_ack_id = None
                        output = dict(playback_id=playback_id, action=event.get('action'),
                                      turn_index=index,
                                      announced_bytes=size, received_bytes=0, binary_frames=0,
                                      audio_start_ms=elapsed(), status='receiving')
                        report['outputs'].append(output)
                        outputs_by_id[playback_id] = output
                        incoming = {**output, 'chunks': [], 'output': output}
                    elif kind == 'audio_stop':
                        playback_id = event.get('playback_id')
                        if playback_id in acknowledgements:
                            acknowledgements[playback_id].cancel()
                        output = outputs_by_id.get(playback_id)
                        if output is not None:
                            output['status'] = 'interrupted'
                            output['interrupted_ms'] = elapsed()
                        if incoming is not None and incoming['playback_id'] == playback_id:
                            output.update(received_bytes=incoming['received_bytes'], binary_frames=incoming['binary_frames'])
                            incoming = None
                    elif kind == 'audio_end':
                        if incoming is None or incoming['playback_id'] != event.get('playback_id'):
                            raise TransportError('audio_end did not match the active audio body.')
                        raw = b''.join(incoming['chunks'])
                        output = incoming['output']
                        output.update(received_bytes=len(raw), binary_frames=incoming['binary_frames'])
                        if len(raw) != incoming['announced_bytes']:
                            raise TransportError('Response audio was truncated or incomplete.')
                        _, audio_metadata = verified_wav(raw)
                        output.update(audio_metadata)
                        path = output_dir / f"reply_{len(report['outputs']) - 1:02d}.wav"
                        path.write_bytes(raw)
                        output.update(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                      audio_ready_ms=elapsed(), status='verified',
                                      device_playback_tested=False, ack_kind='simulated_duration_after_verified_wav')
                        if output['action'] != 'greeting':
                            mark('first_reply_audio_ready')
                            if output['turn_index'] is not None:
                                mark_turn(output['turn_index'], 'first_reply_audio_ready')
                        playback_id = incoming['playback_id']
                        incoming = None
                        acknowledgements[playback_id] = group.create_task(acknowledge(playback_id, output))
                    elif kind == 'error':
                        raise TransportError('Server error: ' + str(event.get('message', 'unspecified failure')))
                    elif kind == 'done':
                        report['server_status'] = event.get('status')
                        report['final_state'] = event.get('slots')
                        mark('server_done')
                        server_done.set()
                        if manifest:
                            # The server can finish while the local ACK send is
                            # still draining. Resolve only those already-sent
                            # completions before deciding a turn was skipped.
                            pending_acks = [task for playback_id, task in acknowledgements.items()
                                            if not task.done() and outputs_by_id[playback_id]['status'] == 'simulated_ack_sending']
                            if pending_acks:
                                await asyncio.gather(*pending_acks)
                            final = turn_records[-1]
                            natural_terminal = (
                                final.get('reply_action') == 'accepted' and event.get('status') in {'demo_completed', 'completed'} or
                                final.get('reply_action') == 'handoff' and event.get('status') == 'handoff')
                            if final.get('reply_action') in {'accepted', 'handoff'} and not natural_terminal:
                                raise TransportError('Terminal reply did not match the final server status.')
                            if (natural_terminal and final['source_complete'] and not final['tail_complete'] and
                                    all(record['status'] == 'completed' for record in turn_records[:-1])):
                                # The server no longer needs conservative tail
                                # silence once its final reply has completed.
                                final['tail_truncated_by_terminal'] = True
                                complete_turn(len(turns) - 1)
                        if manifest and any(record['status'] != 'completed' for record in turn_records):
                            raise TransportError('Server finished early; remaining supplied turns were not completed.')
                        if not reply_done.is_set():
                            raise TransportError('Server finished before a verified reply completed.')
                        return
                if not server_done.is_set():
                    raise TransportError('WebSocket closed without a final session result.')

            async def send_input():
                await started.wait()
                offsets = [0 for _ in turns]
                silent_tails = [0 for _ in turns]
                pacer = FramePacer(frame_ms / 1000)
                first_sends = [None for _ in turns]
                reset_starts = [None for _ in turns]
                pacing_template = {
                    'policy': 'cumulative_deadlines_with_one_frame_stall_reset',
                    'interval_ms': frame_ms, 'reset_threshold_ms': frame_ms,
                    'resets': 0, 'source_frames_sent': 0,
                    'source_send_interval_ms': None,
                    'source_expected_send_interval_ms': None,
                    'source_mean_send_interval_ms': None,
                    'source_padded_duration_ms': 0,
                    'interval_definition': 'First to last successful source-frame send, excluding the final frame duration; socket sends are not device capture or playback.',
                }
                for record in turn_records:
                    record['pacing'] = dict(pacing_template)
                report['pacing'] = dict(pacing_template)
                report['pacing']['source_timing_turn_index'] = 0
                while not stopping.is_set() and not server_done.is_set():
                    # A completed reply may stop the session while this task
                    # is asleep. Never send the next frame after that boundary.
                    if not await pacer.wait_next(lambda: stopping.is_set() or server_done.is_set()):
                        return
                    report['pacing']['resets'] = pacer.resets
                    report['send_max_lag_ms'] = round(pacer.max_lag_seconds * 1000, 3)
                    index = active_turn
                    if index is not None:
                        if reset_starts[index] is None:
                            reset_starts[index] = pacer.resets
                        turn_records[index]['pacing']['resets'] = pacer.resets - reset_starts[index]
                    pcm = turns[index]['pcm'] if index is not None else b''
                    source_frame = index is not None and offsets[index] < len(pcm)
                    if source_frame:
                        mark_turn(index, 'input_audio_started')
                        part = pcm[offsets[index]:offsets[index] + frame_bytes]
                        offsets[index] += len(part)
                        frame = part.ljust(frame_bytes, b'\0')
                    else:
                        frame = bytes(frame_bytes)
                    try:
                        await socket.send(frame)
                    except ConnectionClosedOK:
                        if reply_done.is_set() and (stopping.is_set() or server_done.is_set()):
                            return
                        raise
                    report['frames_sent'] += 1
                    if source_frame:
                        sent_at = time.perf_counter()
                        if first_sends[index] is None:
                            first_sends[index] = sent_at
                        pacing = turn_records[index]['pacing']
                        source_frames = pacing['source_frames_sent'] + 1
                        source_interval = (sent_at - first_sends[index]) * 1000
                        pacing.update(
                            source_frames_sent=source_frames,
                            source_first_sent_ms=round((first_sends[index] - started_at) * 1000, 3),
                            source_last_sent_ms=round((sent_at - started_at) * 1000, 3),
                            source_send_interval_ms=round(source_interval, 3),
                            source_expected_send_interval_ms=(source_frames - 1) * frame_ms,
                            source_mean_send_interval_ms=(round(source_interval / (source_frames - 1), 3)
                                                          if source_frames > 1 else None),
                            source_padded_duration_ms=source_frames * frame_ms,
                        )
                        if index == 0:
                            report['pacing'].update({key: value for key, value in pacing.items() if key != 'resets'})
                    if source_frame and offsets[index] == len(pcm):
                        mark_turn(index, 'input_source_end_sent')
                        turn_records[index].update(source_complete=True, status='waiting')
                    elif index is not None and offsets[index] == len(pcm) and not turn_input_done[index].is_set():
                        silent_tails[index] += 1
                        turn_records[index].update(tail_frames_sent=silent_tails[index],
                                                   tail_ms_sent=silent_tails[index] * frame_ms)
                        if silent_tails[index] >= trailing_frames:
                            mark_turn(index, 'input_and_trailing_silence_complete')
                            turn_records[index]['tail_complete'] = True
                            turn_input_done[index].set()
                            complete_turn(index)

            async def finish(sender):
                nonlocal active_turn
                await greeting_done.wait()
                for index in range(len(turns)):
                    if server_done.is_set():
                        raise TransportError('Server finished before all supplied turns were played.')
                    active_turn = index
                    turn_records[index]['status'] = 'sending'
                    await turn_finished[index].wait()
                active_turn = None
                stopping.set()
                # Join the PCM sender before asking the server to close. This
                # also cancels an in-flight paced wait or websocket send.
                sender.cancel()
                try:
                    await sender
                except asyncio.CancelledError:
                    pass
                if not server_done.is_set():
                    if manifest and turn_records[-1].get('reply_action') in {'accepted', 'handoff'}:
                        # The completion event precedes natural DONE. A stop
                        # here could race the server into an interrupted result.
                        await asyncio.wait_for(server_done.wait(), 10)
                    else:
                        mark('stop_sent')
                        await socket.send(json.dumps({'type': 'stop'}))
                        await asyncio.wait_for(server_done.wait(), 10)

            async with asyncio.TaskGroup() as group:
                group.create_task(receive(group))
                sender = group.create_task(send_input(), name='voice_transport_input')
                group.create_task(finish(sender))


def error_messages(exc):
    if isinstance(exc, BaseExceptionGroup):
        return [message for child in exc.exceptions for message in error_messages(child)]
    return [f'{type(exc).__name__}: {str(exc) or "operation timed out"}']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', help='Configured demo task ID; required with --audio and must match a manifest if supplied.')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--audio', type=Path, help='Original single-WAV diagnostic (use --check for local validation only).')
    inputs.add_argument('--manifest', type=Path, help='JSON domain and 1–5 supplied WAV turns; validation only unless --execute.')
    parser.add_argument('--source-kind', choices=['recorded', 'synthetic_diagnostic'], help='Required with --audio; manifests label each turn.')
    parser.add_argument('--execute', action='store_true', help='Explicitly execute a manifest against the local demo; never generates input audio.')
    parser.add_argument('--timeout-seconds', type=float, default=180,
                        help='Total diagnostic timeout, at most 180 seconds.')
    parser.add_argument('--check', action='store_true', help='Validate input/config locally without opening a connection.')
    args = parser.parse_args()
    try:
        if not 0 < args.timeout_seconds <= 180:
            raise TransportError('Timeout must be greater than zero and at most 180 seconds.')
        if args.manifest:
            from bench.voice_manifest import load_manifest
            if args.source_kind is not None:
                raise TransportError('A manifest requires source_kind on each turn, not --source-kind.')
            manifest = load_manifest(args.manifest, timeout_seconds=args.timeout_seconds)
            if args.domain is not None and args.domain != manifest['domain']:
                raise TransportError('--domain does not match the manifest domain.')
            args.domain = manifest['domain']
            turns = manifest['turns']
        else:
            if args.domain is None or args.source_kind is None:
                raise TransportError('--audio requires --domain and --source-kind.')
            source = args.audio.resolve(strict=True)
            raw = source.read_bytes()
            pcm, audio = verified_wav(raw)
            if audio['duration_seconds'] >= args.timeout_seconds:
                raise TransportError('Input duration must be shorter than the total timeout.')
        from bench.voice_manifest import load_demo_domain
        config = load_demo_domain(args.domain)
        if args.manifest and args.timeout_seconds > config['engine']['max_session_ms'] / 1000:
            raise TransportError('Manifest timeout must not exceed the configured demo session limit.')
    except (OSError, ValueError, RuntimeError) as exc:
        print('NOT RUN:', exc)
        return 2
    if args.check or (args.manifest and not args.execute):
        if args.manifest:
            print(json.dumps({'status': 'manifest_validated_only', 'network_used': False,
                              'domain': args.domain, 'manifest': str(args.manifest.resolve()),
                              'turns': [{key: value for key, value in turn.items() if key != 'pcm'} for turn in turns],
                              'timeout_seconds': args.timeout_seconds,
                              'evaluation_eligible': False, 'expectations_kind': 'diagnostic_unreviewed',
                              'bounds_kind': 'input_and_session_time_only; not a hard provider-request cap',
                              'execute_required': True}, ensure_ascii=False))
        else:
            print(json.dumps({'status': 'input_validated_only', 'network_used': False,
                              'source': str(source), 'source_kind': args.source_kind, **audio}, ensure_ascii=False))
        return 0
    output_dir = ROOT / 'bench/results' / f'voice_transport_{uuid4().hex}'
    output_dir.mkdir(parents=True)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), domain=args.domain,
                  purpose='local_supplied_audio_transport_diagnostic', evaluation_eligible=False,
                  device_playback_tested=False, microphone_capture_tested=False, webaudio_tested=False,
                  playback_ack='simulated after WAV verification and a duration wait; no audio output device',
                  human_listening_verdict='not collected', provider_retries_by_harness=0,
                  provider_switches_by_harness=0, timeout_seconds=args.timeout_seconds,
                  events=[], outputs=[], timings_ms={}, frames_sent=0, status='running', errors=[])
    if args.manifest:
        report['manifest'] = str(args.manifest.resolve())
    else:
        report.update(source=str(source), source_kind=args.source_kind,
                      source_sha256=hashlib.sha256(raw).hexdigest(), input_audio=audio)
    started_at = time.perf_counter()
    try:
        if args.manifest:
            asyncio.run(run_manifest_transport(args, turns, config, report, output_dir))
        else:
            asyncio.run(run_transport(args, pcm, config, report, output_dir))
        report['status'] = 'transport_verified'
    except Exception as exc:
        report['status'] = 'failed'
        report['errors'] = error_messages(exc)
    except KeyboardInterrupt:
        report['status'] = 'interrupted'
        report['errors'] = ['Interrupted by the operator.']
    finally:
        report['elapsed_ms'] = round((time.perf_counter() - started_at) * 1000, 3)
        report_path = output_dir / 'report.json'
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(json.dumps({'status': report['status'], 'report': str(report_path),
                          'errors': report['errors'], 'device_playback_tested': False}, ensure_ascii=False))
    return 0 if report['status'] == 'transport_verified' else 1


if __name__ == '__main__':
    sys.exit(main())
