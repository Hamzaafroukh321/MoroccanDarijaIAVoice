"""Explicit one-input local MoulSot voice diagnostic; importing performs no work."""

import asyncio
from contextlib import AsyncExitStack
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def _paced_frame_ready(pacer, completions):
    """A completion or error during pacing must not admit another input frame."""
    return await pacer.wait_next(lambda: not completions.empty()) and completions.empty()


async def run_voice_probe(audio):
    """Run explicitly against an already started, guarded local llama-server.

    Returns diagnostic evidence on success or failure, saving full reply WAVs.
    The caller owns the upstream server; this helper never starts or stops it.
    """
    from dotenv import load_dotenv
    import httpx
    from bench.voice_transport import FramePacer, verified_wav
    from engine.config import load_config
    from engine.demo import DemoVoice, demo_config
    from engine.endpointing import SileroVAD
    from engine.pipeline import VoiceSession
    from engine.router import Router
    from engine.stt import SpeechToText

    source = Path(audio).resolve()
    # Reject oversized input before reading it or allocating any provider client.
    if source.stat().st_size > 705536:
        raise ValueError('Supply one WAV of at most 20 seconds and 705536 bytes.')
    raw = source.read_bytes()
    spec = importlib.util.spec_from_file_location('local_moulsot_probe_bridge', Path(__file__).with_name('bridge.py'))
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    bridge.validate_wav(raw)
    pcm, audio_info = verified_wav(raw)
    load_dotenv(ROOT / '.env', override=False)
    config = demo_config(load_config(ROOT / 'configs/pizza.json'))
    if config['demo']['tts_provider'] != 'darija_xtts':
        raise ValueError('This diagnostic requires the selected darija_xtts voice.')
    config['stt']['moulsot_protocol'] = 'json'
    config['stt']['timeout_ms'] = 32000
    directory = ROOT / 'bench/results' / ('local_pipeline_' + uuid4().hex)
    directory.mkdir(parents=True)
    started = time.perf_counter()
    report = {'kind': 'local_voice_session_diagnostic', 'evaluation_eligible': False,
              'source': {'path': str(source), 'sha256': hashlib.sha256(raw).hexdigest(), **audio_info},
              'bounds': {'input_files': 1, 'max_input_seconds': 20, 'timeout_seconds': 55,
                         'timeout_scope': 'Session initialization and input/reply wait; resource cleanup follows.',
                         'provider_calls': 'Existing VoiceSession/router retry limits; no hard provider-call cap.'},
              'limitations': ['Saved recording, not a live microphone or browser/TCP test.',
                              'Immediate playback ACK is simulated; no device playback or listener validation.',
                              'No human transcript, intended task or accuracy/completion claim is supplied.',
                              'ASR uses quantized MoulSot locally; Groq router and uncached XTTS use real remote providers.'],
              'status': 'running', 'events': [], 'audio_outputs': [], 'bridge_responses': [],
              'source_frames_sent': 0, 'tail_frames_sent': 0, 'source_complete': False}
    voice = None
    current = None
    completed_ids = set()
    completions = asyncio.Queue()
    endpoint_count = 0
    pacer = FramePacer(.032)

    def elapsed():
        return round((time.perf_counter() - started) * 1000, 3)

    async def emit(event):
        nonlocal current, endpoint_count
        report['events'].append({'at_ms': elapsed(), **deepcopy(event)})
        kind = event['type']
        if kind == 'endpoint':
            endpoint_count += 1
            if endpoint_count > 1:
                raise RuntimeError('More than one endpoint; single-input diagnostic stopped.')
        elif kind == 'audio_start':
            if current is not None:
                raise RuntimeError('Overlapping audio output.')
            current = {'playback_id': event['playback_id'], 'action': event['action'],
                       'announced_bytes': event['bytes'], 'raw': bytearray()}
        elif kind == 'audio_end':
            if current is None or current['playback_id'] != event['playback_id']:
                raise RuntimeError('Audio end does not match the current output.')
            output = current
            wav = bytes(output.pop('raw'))
            if len(wav) != output['announced_bytes']:
                raise RuntimeError('Output audio is incomplete.')
            _, metadata = verified_wav(wav)
            path = directory / f'reply_{len(report["audio_outputs"]):02d}.wav'
            path.write_bytes(wav)
            output.update(path=str(path), received_bytes=len(wav), sha256=hashlib.sha256(wav).hexdigest(),
                          audio=metadata, ack='simulated_immediate', completed=False)
            report['audio_outputs'].append(output)
            current = None
            await voice.playback_finished(event['playback_id'])
        elif kind == 'playback_complete':
            if event['playback_id'] in completed_ids:
                return
            output = next((item for item in report['audio_outputs']
                           if item['playback_id'] == event['playback_id']), None)
            if output is None or output['action'] != event['action']:
                raise RuntimeError('Completion lacks matching verified audio and ACK.')
            completed_ids.add(event['playback_id'])
            output.update(completed=True, completion_at_ms=elapsed())
            completions.put_nowait(deepcopy(event))
        elif kind == 'error':
            completions.put_nowait(deepcopy(event))

    async def emit_audio(frame):
        if current is None:
            raise RuntimeError('Audio bytes arrived without an output header.')
        if len(current['raw']) + len(frame) > min(current['announced_bytes'], 8 * 1024 * 1024):
            raise RuntimeError('Output exceeded its announced or diagnostic byte limit.')
        current['raw'].extend(frame)

    async def next_completion():
        event = await completions.get()
        if event['type'] == 'error':
            raise RuntimeError(event.get('message', 'VoiceSession error.'))
        return event

    async def record_bridge(response):
        await response.aread()
        if response.status_code == 200:
            payload = response.json()
            report['bridge_responses'].append({'at_ms': elapsed(), **payload})
        else:
            report['bridge_responses'].append({'at_ms': elapsed(), 'status_code': response.status_code})

    try:
        async with AsyncExitStack() as resources:
            async with asyncio.timeout(55):
                vad = await asyncio.to_thread(SileroVAD, config)
                application = bridge.create_app()
                await resources.enter_async_context(application.router.lifespan_context(application))
                client = await resources.enter_async_context(httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=application), base_url='http://bridge',
                    timeout=32, trust_env=False, event_hooks={'response': [record_bridge]}))
                stt = SpeechToText(config, ROOT, primary='moulsot', fallback=False,
                                   endpoint='http://bridge/transcribe', client=client)
                resources.push_async_callback(stt.close)
                router = Router(config, ROOT)
                resources.push_async_callback(router.close)
                bank = DemoVoice(config, ROOT)
                resources.push_async_callback(bank.close)
                voice = VoiceSession(config, ROOT, stt, router, bank, vad, emit, emit_audio)
                resources.push_async_callback(voice.close)
                await voice.start()
                greeting = await next_completion()
                if greeting['action'] != 'greeting':
                    raise RuntimeError('Expected the greeting before supplied audio.')
                report['greeting_complete_ms'] = elapsed()
                frame_bytes = voice.detector.frame_bytes
                tail_frames = math.ceil(max(config['endpointing']['base_silence_ms'],
                                            config['endpointing']['max_wait_ms']) / 32) + 2
                for offset in range(0, len(pcm), frame_bytes):
                    if not completions.empty():
                        raise RuntimeError('Reply or error arrived before the supplied WAV finished.')
                    if not await _paced_frame_ready(pacer, completions):
                        raise RuntimeError('Reply or error arrived before the supplied WAV finished.')
                    report.setdefault('source_first_frame_ms', elapsed())
                    await voice.feed(pcm[offset:offset + frame_bytes].ljust(frame_bytes, b'\0'))
                    report['source_last_frame_ms'] = elapsed()
                    report['source_frames_sent'] += 1
                report['source_complete'] = True
                report['source_complete_ms'] = elapsed()
                for _ in range(tail_frames):
                    if not completions.empty():
                        break
                    if not await _paced_frame_ready(pacer, completions):
                        break
                    await voice.feed(bytes(frame_bytes))
                    report['tail_frames_sent'] += 1
                reply = await next_completion()
                report['reply_completion'] = reply
                if endpoint_count != 1 or len(voice.turns) != 1 or voice.turns[0]['provider'] != 'moulsot':
                    raise RuntimeError('Expected exactly one endpoint and actual MoulSot transcript.')
                report['status'] = 'input_and_reply_verified'
                report['reply_complete_ms'] = elapsed()
    except asyncio.CancelledError:
        report.update(status='cancelled', error_type='CancelledError')
        raise
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    finally:
        report.update(elapsed_ms=elapsed(), endpoint_count=endpoint_count,
                      pacing_resets=pacer.resets, pacing_max_lag_ms=pacer.max_lag_seconds * 1000)
        if voice is not None:
            try:
                session = voice.save()
                report['session_report'] = str(ROOT / config['runtime']['results_dir'] / f'demo_session_{voice.session_id}.json')
                report['session'] = {key: session.get(key) for key in (
                    'session_id', 'status', 'state', 'confirmed', 'turns', 'outputs', 'stt_calls',
                    'router_calls', 'tts_calls', 'transcript_requests', 'pending_clarification', 'pending_proposal')}
            except Exception as exc:
                report['session_save_error'] = {'type': type(exc).__name__, 'message': str(exc)}
                if report['status'] == 'input_and_reply_verified':
                    report['status'] = 'evidence_save_failed'
        report['report_path'] = str(directory / 'report.json')
        (directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report
