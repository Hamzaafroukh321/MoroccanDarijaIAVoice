"""Start one isolated loopback ASR server, measure one request, always stop it."""
import argparse
import asyncio
import base64
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import httpx

from prepare import DEST, ROOT
from probe import available_ram_mib, free_gpu_mib


async def engine_request(audio):
    """Real engine adapter through in-process bridge to real loopback model API."""
    import wave
    sys.path.insert(0, str(ROOT))
    from engine.config import load_config
    from engine.stt import SpeechToText
    from bridge import create_app
    config = load_config(ROOT / 'configs/clinic.json')
    config['stt']['moulsot_protocol'] = 'json'
    app = create_app()
    replies = []

    async def save_reply(response):
        await response.aread()
        replies.append(response.json())

    with wave.open(str(audio), 'rb') as wav:
        pcm = wav.readframes(wav.getnframes())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     event_hooks={'response': [save_reply]}) as client:
            stt = SpeechToText(config, ROOT, primary='moulsot', endpoint='http://bridge/transcribe',
                               client=client, fallback=False, hf_token='')
            started = time.perf_counter()
            transcript = await stt.transcribe(pcm)
            return {'elapsed_ms': (time.perf_counter() - started) * 1000,
                    'result': asdict(transcript), 'calls': stt.calls, 'bridge_replies': replies,
                    'transport': 'real SpeechToText -> ASGI bridge -> TCP llama-server', 'fallback': False}


def run(audio, gpu_projector=False, engine_bridge=False, voice_pipeline=False):
    audio = Path(audio).resolve(strict=True)
    assert (DEST / 'verified-assets.json').is_file(), 'Prepare verified assets first'
    executable, = (DEST / 'runtime').rglob('llama-server.exe')
    with socket.socket() as check:
        check.bind(('127.0.0.1', 8011))
    env = os.environ.copy()
    env['PATH'] = os.pathsep.join(sorted({str(p.parent) for p in (DEST / 'runtime').rglob('*.dll')})) + os.pathsep + env['PATH']
    command = [str(executable), '-m', str(DEST / 'models/moulsot.v0.3.Q4_K_M.gguf'),
               '--mmproj', str(DEST / 'models/moulsot.v0.3.mmproj-Q8_0.gguf'),
               '-n', '128', '-c', '2048', '--temp', '0', '-ngl', '99', '--no-warmup',
               '--no-mmproj-offload', '-t', '4', '-b', '128', '-ub', '64', '-np', '1',
               '--host', '127.0.0.1', '--port', '8011', '--log-verbosity', '5',
               '--cors-origins', 'http://127.0.0.1:8011']
    if gpu_projector:
        command.remove('--no-mmproj-offload')
    if engine_bridge or voice_pipeline:
        from bridge import MAX_OUTPUT_TOKENS
        command[command.index('-n') + 1] = str(MAX_OUTPUT_TOKENS)
    prefix = ROOT / 'bench/results' / time.strftime('local_moulsot_warm_%Y%m%d_%H%M%S')
    report = {'purpose': 'One model-resident local ASR request; not native accuracy', 'command': command,
              'audio_sha256': hashlib.sha256(audio.read_bytes()).hexdigest(), 'requests': [], 'samples': [],
              'deadline_seconds': 120, 'production_changed': False}
    report['before'] = {'free_ram_mib': available_ram_mib(), 'free_gpu_mib': free_gpu_mib()}
    if report['before']['free_ram_mib'] < 2300 or report['before']['free_gpu_mib'] < 1700:
        report['status'] = 'not_started_resource_headroom'
    else:
        started = time.perf_counter()
        stop = threading.Event()
        with prefix.with_suffix('.stdout.log').open('wb') as stdout, prefix.with_suffix('.stderr.log').open('wb') as stderr:
            process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            report['pid'] = process.pid

            def monitor():
                while not stop.is_set() and process.poll() is None:
                    try:
                        sample = {'elapsed_seconds': time.perf_counter() - started,
                                  'free_ram_mib': available_ram_mib(), 'free_gpu_mib': free_gpu_mib()}
                        report['samples'].append(sample)
                        if sample['elapsed_seconds'] > 120 or sample['free_ram_mib'] < 768 or sample['free_gpu_mib'] < 256:
                            report['guard_stop'] = sample
                            process.kill()
                            return
                    except Exception as exc:
                        report['guard_error'] = type(exc).__name__
                        process.kill()
                        return
                    stop.wait(1)

            thread = threading.Thread(target=monitor)
            thread.start()
            try:
                with httpx.Client(base_url='http://127.0.0.1:8011', timeout=45, trust_env=False) as client:
                    while time.perf_counter() - started < 60:
                        if process.poll() is not None:
                            raise RuntimeError('Server exited during startup')
                        try:
                            if client.get('/health', timeout=2).status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        time.sleep(.5)
                    else:
                        raise TimeoutError('Startup deadline')
                    report['startup_ms'] = (time.perf_counter() - started) * 1000
                    if voice_pipeline:
                        from pipeline_probe import run_voice_probe
                        pipeline = asyncio.run(run_voice_probe(audio))
                        report['requests'].append(pipeline)
                        report['status'] = pipeline['status']
                        return
                    if engine_bridge:
                        report['requests'].append(asyncio.run(engine_request(audio)))
                        report['status'] = 'engine_transcribed'
                        return
                    payload = {'messages': [{'role': 'user', 'content': [
                        {'type': 'text', 'text': 'a'},
                        {'type': 'input_audio', 'input_audio': {'data': base64.b64encode(audio.read_bytes()).decode(), 'format': 'wav'}}]}],
                        'max_tokens': 128, 'temperature': 0, 'cache_prompt': False, 'stream': False}
                    request_started = time.perf_counter()
                    response = client.post('/v1/chat/completions', json=payload)
                    report['requests'].append({'elapsed_ms': (time.perf_counter() - request_started) * 1000,
                                               'http_status': response.status_code, 'response': response.json()})
                    response.raise_for_status()
                    report['status'] = 'transcribed'
            except Exception as exc:
                report['status'] = 'failed'
                report['error'] = str(exc)
            finally:
                stop.set()
                thread.join(timeout=7)
                if process.poll() is None:
                    process.kill()
                report['exit_code_after_cleanup'] = process.wait(timeout=10)
                report['total_elapsed_ms'] = (time.perf_counter() - started) * 1000
                prefix.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                print(json.dumps({'report': str(prefix.with_suffix('.json')), 'status': report['status']}, indent=2))
        return
    prefix.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(prefix.with_suffix('.json')), 'status': report['status']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--gpu-projector', action='store_true', help='One explicit GPU encoder comparison')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--engine-bridge', action='store_true', help='Measure existing engine ASR adapter through optional bridge')
    group.add_argument('--voice-pipeline', action='store_true', help='One supplied WAV through real conversation providers; simulated playback ACK')
    args = parser.parse_args()
    run(args.audio, args.gpu_projector, args.engine_bridge, args.voice_pipeline)
