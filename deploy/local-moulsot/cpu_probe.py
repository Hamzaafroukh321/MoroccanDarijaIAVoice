"""One guarded CPU-only MoulSot comparison; default is local preflight only."""

import argparse
import base64
import hashlib
import json
import os
import socket
import threading
import time
from pathlib import Path
from uuid import uuid4
import wave
import zipfile

import httpx

from prepare import ASSETS, DEST, ROOT, digest
from probe import available_ram_mib
from bridge import parse_transcription, validate_wav
from windows_job import WindowsJob


def verified_runtime():
    if not (DEST / 'verified-assets.json').is_file():
        raise ValueError('Verified asset manifest is required.')
    for name, size, sha, _ in ASSETS:
        path = DEST / ('models' if name.endswith('.gguf') else 'archives') / name
        if not path.is_file() or path.stat().st_size != size or digest(path) != sha:
            raise ValueError(f'Missing or unverified pinned asset: {name}')
        if name.endswith('.zip'):
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if member.filename.endswith('.dll') or member.filename.endswith('llama-server.exe'):
                        extracted = DEST / 'runtime' / name.removesuffix('.zip') / member.filename
                        with archive.open(member) as stream:
                            expected = hashlib.file_digest(stream, 'sha256').hexdigest()
                        if not extracted.is_file() or digest(extracted) != expected:
                            raise ValueError('Extracted runtime does not match its pinned archive.')
    executable, = (DEST / 'runtime').rglob('llama-server.exe')
    return executable


def run(execute=False):
    started = time.perf_counter()
    directory = ROOT / 'bench/results' / ('local_moulsot_cpu_' + uuid4().hex)
    directory.mkdir(parents=True)
    report = {'kind': 'one_cpu_only_moulsot_comparison', 'evaluation_eligible': False,
              'source_kind': 'synthetic_diagnostic', 'deadline_seconds': 120,
              'min_start_ram_mib': 3072, 'min_running_ram_mib': 768,
              'port': 8013, 'other_services_changed': False, 'requests': [], 'memory_samples': [],
              'status': 'preflight', 'native_review': False,
              'limitations': ['One short synthetic correction, not native accuracy or a sustained latency benchmark.',
                              'The existing GPU service and other applications remain running; CPU timing includes this contention.']}
    stop = threading.Event()
    monitor_thread = None
    job = None
    process = None
    try:
        with socket.socket() as check:
            check.bind(('127.0.0.1', 8013))
        report['initial_free_ram_mib'] = available_ram_mib()
        if report['initial_free_ram_mib'] < 3072:
            report['status'] = 'not_started_insufficient_ram'
            return report
        executable = verified_runtime()
        source = ROOT / 'bench/results/clinic_sequence_20260905/turn_2_synthetic.wav'
        raw = source.read_bytes()
        validate_wav(raw)
        sha = hashlib.sha256(raw).hexdigest()
        if sha != '3bd55e3191f458d4aca45adc753e9b434658e4bd45b042ba9e6e169e7bac1ad7':
            raise ValueError('The supplied correction differs from its saved provenance.')
        with wave.open(str(source), 'rb') as wav:
            seconds = wav.getnframes() / wav.getframerate()
        report['source'] = {'path': str(source), 'sha256': sha, 'seconds': seconds,
                            'provenance': str(source.with_name('provenance.json'))}
        report['artifacts_verified'] = True
        command = [str(executable), '-m', str(DEST / 'models/moulsot.v0.3.Q4_K_M.gguf'),
                   '--mmproj', str(DEST / 'models/moulsot.v0.3.mmproj-Q8_0.gguf'),
                   '--device', 'none', '-ngl', '0', '--no-mmproj-offload', '-t', '4',
                   '-n', '128', '-c', '2048', '--temp', '0', '--no-warmup', '-b', '128', '-ub', '64',
                   '-np', '1', '--host', '127.0.0.1', '--port', '8013', '--log-verbosity', '3',
                   '--cors-origins', 'http://127.0.0.1:8013']
        report['command'] = command
        if not execute:
            report['status'] = 'preflight_ready'
            return report
        # Recheck immediately before allocation, after reading/hashing artifacts.
        report['launch_free_ram_mib'] = available_ram_mib()
        if report['launch_free_ram_mib'] < 3072:
            report['status'] = 'not_started_insufficient_ram'
            return report
        with socket.socket() as check:
            check.bind(('127.0.0.1', 8013))
        if time.perf_counter() - started >= 120:
            raise TimeoutError('Preflight exhausted the total deadline.')
        environment = os.environ.copy()
        environment['PATH'] = os.pathsep.join(sorted({str(path.parent) for path in (DEST / 'runtime').rglob('*.dll')})) + os.pathsep + environment.get('PATH', '')
        job = WindowsJob()
        process = job.spawn(command, cwd=ROOT, env=environment,
                            stdout=directory / 'stdout.log', stderr=directory / 'stderr.log')
        report['pid'] = process.pid
        report['spawn_ms'] = (time.perf_counter() - started) * 1000

        def monitor():
            while not stop.is_set():
                try:
                    sample = {'at_ms': (time.perf_counter() - started) * 1000, 'free_ram_mib': available_ram_mib()}
                    report['memory_samples'].append(sample)
                    if sample['at_ms'] >= 120000 or sample['free_ram_mib'] < 768:
                        report['guard_stop'] = sample
                        job.close()
                        return
                except Exception as exc:
                    report['guard_error_type'] = type(exc).__name__
                    job.close()
                    return
                stop.wait(.5)

        monitor_thread = threading.Thread(target=monitor, name='cpu-probe-resource-guard', daemon=True)
        monitor_thread.start()
        print(json.dumps({'status': 'cpu_probe_started', 'pid': process.pid, 'report_directory': str(directory)}), flush=True)
        with httpx.Client(base_url='http://127.0.0.1:8013', timeout=2, trust_env=False, follow_redirects=False) as client:
            while time.perf_counter() - started < 90:
                if process.poll() is not None:
                    raise RuntimeError('CPU model exited during startup.')
                try:
                    if client.get('/health', timeout=1).status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(.25)
            else:
                raise TimeoutError('CPU startup did not finish within 90 seconds total.')
            report['ready_ms'] = (time.perf_counter() - started) * 1000
            models_response = client.get('/v1/models')
            models_response.raise_for_status()
            models = models_response.json()
            identities = [item.get('id') for item in models.get('data', []) if isinstance(item, dict)]
            if len(identities) != 1 or not isinstance(identities[0], str) or identities[0].replace('\\', '/').rsplit('/', 1)[-1] != 'moulsot.v0.3.Q4_K_M.gguf':
                raise ValueError('CPU server did not advertise the expected model.')
            report['upstream_models'] = models
            payload = {'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': 'a'}, {'type': 'input_audio', 'input_audio': {
                    'data': base64.b64encode(raw).decode('ascii'), 'format': 'wav'}}]}],
                'max_tokens': 128, 'temperature': 0, 'cache_prompt': False, 'stream': False}
            request_started = time.perf_counter()
            request = {'started_at_ms': (request_started - started) * 1000, 'max_tokens': 128}
            report['requests'].append(request)
            try:
                remaining = 120 - (time.perf_counter() - started)
                if remaining <= 0:
                    raise TimeoutError('Total deadline reached before inference.')
                with client.stream('POST', '/v1/chat/completions', json=payload, timeout=remaining) as response:
                    request['http_status'] = response.status_code
                    response.raise_for_status()
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        if len(body) + len(chunk) > 65536:
                            raise ValueError('CPU response exceeded the bounded response size.')
                        body.extend(chunk)
                request['response'] = json.loads(body)
                request['transcription'] = parse_transcription(request['response'])
                report['status'] = 'cpu_transcribed'
            finally:
                request['elapsed_ms'] = (time.perf_counter() - request_started) * 1000
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    finally:
        stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=12)
        if job is not None:
            job.close()
        if process is not None:
            report['exit_code_after_cleanup'] = process.poll()
        report['elapsed_ms'] = (time.perf_counter() - started) * 1000
        stderr = directory / 'stderr.log'
        if stderr.exists():
            report['device_log_evidence'] = [line for line in stderr.read_text(encoding='utf-8', errors='replace').splitlines()
                if any(term in line for term in ('offloaded', 'model buffer size', 'compute buffer size', 'device =', 'device none'))]
        report['report_path'] = str(directory / 'report.json')
        (directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    result = run(parser.parse_args().execute)
    print(json.dumps({'status': result['status'], 'report': result['report_path']}, indent=2))
