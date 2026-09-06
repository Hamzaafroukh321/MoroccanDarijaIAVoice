"""Supervise the optional local MoulSot demo. Dry preflight unless --start.

All children live in one Windows job. .env is never modified. An optional
one-way hosted recovery starts only the normal engine after a local failure.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
import uuid
import zipfile

import httpx

from prepare import ASSETS, DEST, ROOT, digest
from probe import available_ram_mib, free_gpu_mib
from bridge import MAX_OUTPUT_TOKENS
from engine.asr_provenance import (
    ENV_VAR as RUNTIME_SNAPSHOT_ENV, MODEL_VARIANT, SOURCE_MODEL,
    VERIFICATION, validate_snapshot,
)


PORTS = (8000, 8011, 8012)
STATE = DEST / 'supervisor.json'
STOP = DEST / 'STOP'


def write_status_snapshot(path, payload):
    """Keep readers on complete JSON; tolerate brief Windows sharing locks."""
    temporary = path.with_suffix('.tmp')
    for attempt in range(3):
        try:
            temporary.write_text(payload, encoding='utf-8')
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(.05 * (attempt + 1))


def memory_snapshot(device):
    if device not in {'cpu', 'cuda'}:
        raise ValueError('device must be cpu or cuda')
    memory = {'free_ram_mib': available_ram_mib()}
    if device == 'cuda':
        memory['free_gpu_mib'] = free_gpu_mib()
    return memory


def runtime_memory_exhausted(device, memory):
    return memory['free_ram_mib'] < 768 or (device == 'cuda' and memory['free_gpu_mib'] < 256)


def model_command(executable, device):
    if device not in {'cpu', 'cuda'}:
        raise ValueError('device must be cpu or cuda')
    command = [str(executable), '-m', str(DEST / 'models/moulsot.v0.3.Q4_K_M.gguf'),
               '--mmproj', str(DEST / 'models/moulsot.v0.3.mmproj-Q8_0.gguf'),
               '-n', str(MAX_OUTPUT_TOKENS), '-c', '2048', '--temp', '0',
               '-ngl', '0' if device == 'cpu' else '99', '--no-warmup',
               '-t', '4', '-b', '128', '-ub', '64', '-np', '1',
               '--host', '127.0.0.1', '--port', '8011',
               '--cors-origins', 'http://127.0.0.1:8011']
    if device == 'cpu':
        command += ['--device', 'none', '--no-mmproj-offload']
    return command


def occupied_ports():
    occupied = []
    for port in PORTS:
        with socket.socket() as check:
            try:
                check.bind(('127.0.0.1', port))
            except OSError:
                occupied.append(port)
    return occupied


def preflight(verify=True, device='cuda'):
    if device not in {'cpu', 'cuda'}:
        raise ValueError('device must be cpu or cuda')
    issues = []
    for name, size, sha, _ in ASSETS:
        path = DEST / ('models' if name.endswith('.gguf') else 'archives') / name
        if not path.is_file() or path.stat().st_size != size or (verify and digest(path) != sha):
            issues.append(f'Missing or unverified asset: {name}')
        elif verify and name.endswith('.zip'):
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if member.filename.endswith('.dll') or member.filename.endswith('llama-server.exe'):
                        extracted = DEST / 'runtime' / name.removesuffix('.zip') / member.filename
                        if not extracted.is_file():
                            issues.append(f'Missing extracted runtime file: {member.filename}')
                            continue
                        with archive.open(member) as source:
                            expected = hashlib.file_digest(source, 'sha256').hexdigest()
                        if digest(extracted) != expected:
                            issues.append(f'Extracted runtime differs from verified archive: {member.filename}')
    executables = list((DEST / 'runtime').rglob('llama-server.exe'))
    if len(executables) != 1:
        issues.append('Prepare exactly one verified llama-server runtime.')
    ports = occupied_ports()
    if ports:
        issues.append(f'Ports already occupied: {ports}; do not stop unrelated processes.')
    memory = memory_snapshot(device)
    if device == 'cpu' and memory['free_ram_mib'] < 3072:
        issues.append('Insufficient current headroom: CPU mode requires 3072 MiB RAM free.')
    elif device == 'cuda' and (memory['free_ram_mib'] < 2560 or memory['free_gpu_mib'] < 1900):
        issues.append('Insufficient current headroom: require 2560 MiB RAM and 1900 MiB GPU free.')
    if STOP.exists():
        issues.append('STOP marker exists; remove it explicitly before another run.')
    runtime = json.loads((ROOT / 'configs/pizza.json').read_text(encoding='utf-8'))['runtime']
    if runtime.get('host') != '127.0.0.1' or runtime.get('port') != 8000:
        issues.append('The engine runner must bind 127.0.0.1:8000 for this launcher.')
    result = {'ready': not issues, 'issues': issues, 'memory': memory, 'occupied_ports': ports, 'selected_device': device}
    if verify is True and not issues:
        result.update(verification=VERIFICATION,
                      verified_at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
    return result


def launch_snapshot(check):
    """Use this launch's completed verification, never an inherited claim.

    Called only after the owned runtime has passed its model identity check.
    Selected device records launcher policy, not measured tensor placement.
    """
    if check.get('ready') is not True or check.get('verification') != VERIFICATION:
        return None
    assets = {name: (sha, base) for name, _, sha, base in ASSETS}
    decoder, conversion = assets['moulsot.v0.3.Q4_K_M.gguf']
    projector, _ = assets['moulsot.v0.3.mmproj-Q8_0.gguf']
    runtime, release = assets['llama-b10809-bin-win-cuda-12.4-x64.zip']
    cuda, _ = assets['cudart-llama-bin-win-cuda-12.4-x64.zip']
    return validate_snapshot({
        'schema_version': 1, 'run_id': uuid.uuid4().hex,
        'verified_at': check.get('verified_at'), 'source_model': SOURCE_MODEL,
        'model_variant': MODEL_VARIANT, 'verification': VERIFICATION,
        'selected_device': check.get('selected_device'),
        'conversion_revision': conversion.rstrip('/').rsplit('/', 1)[-1],
        'runtime_release': release.rstrip('/').rsplit('/', 1)[-1],
        'decoder_sha256': decoder, 'projector_sha256': projector,
        'runtime_archive_sha256': runtime, 'cuda_archive_sha256': cuda,
    })


def main(args):
    device = getattr(args, 'device', 'cuda')
    check = preflight(device=device)
    if not args.start or not check['ready']:
        print(json.dumps(check, indent=2))
        return 0 if check['ready'] else 2
    from windows_job import WindowsJob
    prefix = ROOT / 'bench/results' / time.strftime('local_demo_%Y%m%d_%H%M%S')
    state = {'supervisor_pid': os.getpid(), 'status': 'starting', 'mode': 'local', 'selected_device': device,
             'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'children': [],
             'preflight': check, 'env_modified': False, 'recovered_once': False}
    started = time.perf_counter()
    base_env = os.environ.copy()
    base_env.pop(RUNTIME_SNAPSHOT_ENV, None)
    runtime_env = base_env.copy()
    runtime_env['PATH'] = os.pathsep.join(sorted({str(p.parent) for p in (DEST / 'runtime').rglob('*.dll')})) + os.pathsep + base_env.get('PATH', '')
    engine_env = base_env.copy()
    engine_env.update(MOULSOT_PROTOCOL='json', MOULSOT_ENDPOINT='http://127.0.0.1:8012/transcribe', STT_PRIMARY='moulsot')
    model, = (DEST / 'runtime').rglob('llama-server.exe')
    model_args = model_command(model, device)
    bridge_command = [sys.executable, '-X', 'utf8', '-m', 'uvicorn', 'bridge:app',
                      '--app-dir', str(Path(__file__).parent), '--host', '127.0.0.1', '--port', '8012', '--workers', '1']
    # Preserve the engine's tested PCM frame/queue/compression WebSocket settings.
    engine_command = [sys.executable, '-X', 'utf8', '-m', 'engine.server']
    job = None
    processes = []
    publication_failures = set()

    def save():
        state['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        state['elapsed_seconds'] = time.perf_counter() - started
        # Observability must not kill healthy owned services. Each destination is
        # independent; a locked dashboard snapshot can still be recorded in the
        # per-run archive. Retry on the normal loop after a bounded local attempt.
        for label, path in (('current', STATE), ('archive', prefix.with_suffix('.json'))):
            # A successfully published snapshot should already report this
            # destination's recovery; a failed attempt leaves the old file intact.
            state['status_publication_unavailable'] = sorted(publication_failures - {label})
            try:
                write_status_snapshot(path, json.dumps(state, indent=2))
            except OSError as exc:
                state['status_publication_failures'] = state.get('status_publication_failures', 0) + 1
                if label not in publication_failures:
                    print(f'Supervisor {label} status unavailable ({type(exc).__name__}); '
                          'services remain supervised; snapshot may be stale.', file=sys.stderr, flush=True)
                publication_failures.add(label)
            else:
                if label in publication_failures:
                    print(f'Supervisor {label} status publication recovered.', file=sys.stderr, flush=True)
                publication_failures.discard(label)
        state['status_publication_unavailable'] = sorted(publication_failures)

    def spawn(name, command, env):
        stdout = Path(str(prefix) + f'.{name}.stdout.log')
        stderr = Path(str(prefix) + f'.{name}.stderr.log')
        process = job.spawn(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        processes.append(process)
        state['children'].append({'name': name, 'pid': process.pid, 'stdout': str(stdout), 'stderr': str(stderr)})
        save()
        return process

    def wait_ready(url, process, seconds=60):
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            if STOP.exists() or (args.max_seconds and time.perf_counter() - started >= args.max_seconds):
                raise KeyboardInterrupt()
            if process.poll() is not None:
                raise RuntimeError(f'Service exited before readiness: {url}')
            if state['mode'] == 'local':
                if runtime_memory_exhausted(device, memory_snapshot(device)):
                    raise RuntimeError('Resource guard triggered during startup')
            try:
                response = client.get(url, timeout=2)
                if response.status_code == 200:
                    return response
            except httpx.TransportError:
                pass
            time.sleep(.5)
        raise TimeoutError(f'Service readiness deadline: {url}')

    def check_demo(response):
        match = re.search(r'<script id="capture-config" type="application/json">(.*?)</script>', response.text, re.S)
        if match is None:
            raise RuntimeError('Engine page is missing its runtime configuration')
        public = json.loads(match.group(1))
        if not public.get('demo_supported') or public.get('demo_issues') or public.get('demo_tts_provider') != 'darija_xtts':
            raise RuntimeError('Engine demo configuration is not ready for the selected Darija XTTS voice')

    try:
        with httpx.Client(trust_env=False, timeout=3) as client:
            job = WindowsJob()
            model_process = spawn('model', model_args, runtime_env)
            wait_ready('http://127.0.0.1:8011/health', model_process)
            models = client.get('http://127.0.0.1:8011/v1/models').json().get('data', [])
            if len(models) != 1 or models[0]['id'].replace('\\', '/').rsplit('/', 1)[-1] != 'moulsot.v0.3.Q4_K_M.gguf':
                raise RuntimeError('Unexpected local model identity')
            bridge_env = base_env.copy()
            snapshot = launch_snapshot(check)
            if snapshot is not None:
                serialized = json.dumps(snapshot, sort_keys=True, separators=(',', ':'))
                bridge_env[RUNTIME_SNAPSHOT_ENV] = serialized
                engine_env[RUNTIME_SNAPSHOT_ENV] = serialized
                state['local_launch_snapshot'] = snapshot
            bridge_process = spawn('bridge', bridge_command, bridge_env)
            wait_ready('http://127.0.0.1:8012/health', bridge_process, 20)
            engine_process = spawn('engine', engine_command, engine_env)
            check_demo(wait_ready('http://127.0.0.1:8000/?domain=pizza&mode=demo', engine_process, 30))
            check_demo(wait_ready('http://127.0.0.1:8000/?domain=clinic&mode=demo', engine_process, 5))
            state['status'] = 'ready'
            save()
            print(f'Local MoulSot demo ready ({device}) at http://127.0.0.1:8000/?domain=pizza&mode=demo', flush=True)
            while True:
                if STOP.exists() or (args.max_seconds and time.perf_counter() - started >= args.max_seconds):
                    state['status'] = 'stopped'
                    break
                failure = next((f'Owned service exited (PID {p.pid})' for p in processes if p.poll() is not None), None)
                log_bytes = sum(Path(child[key]).stat().st_size for child in state['children']
                                if not child.get('ended')
                                for key in ('stdout', 'stderr') if Path(child[key]).exists())
                state['log_bytes'] = log_bytes
                if log_bytes > 64 * 1024 * 1024:
                    failure = 'Supervisor log limit reached (64 MiB)'
                if state['mode'] == 'local':
                    state['memory'] = memory_snapshot(device)
                    if runtime_memory_exhausted(device, state['memory']):
                        failure = 'Local memory guard triggered'
                if failure:
                    state['failure'] = failure
                    if not args.hosted_fallback or state['recovered_once']:
                        raise RuntimeError(failure)
                    job.close()
                    processes.clear()
                    for child in state['children']:
                        child['ended'] = True
                    # One recovery only, using the existing .env configuration.
                    # No alternate ASR model, inference retries or GPU restart loop.
                    job = WindowsJob()
                    state.update(mode='hosted', status='recovering', recovered_once=True)
                    hosted_env = base_env.copy()
                    hosted_env.pop(RUNTIME_SNAPSHOT_ENV, None)
                    hosted_env['STT_PRIMARY'] = 'moulsot'
                    engine_process = spawn('hosted_engine', engine_command, hosted_env)
                    check_demo(wait_ready('http://127.0.0.1:8000/?domain=pizza&mode=demo', engine_process, 30))
                    state['status'] = 'ready'
                    print('Local service failed; restored existing hosted MoulSot demo once.', flush=True)
                save()
                time.sleep(2)
    except KeyboardInterrupt:
        state['status'] = 'stopped'
    except Exception as exc:
        state.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    finally:
        if job is not None:
            job.close()
        state['shutdown'] = 'owned Windows job terminated; active sessions may not save their final report'
        for child in state['children']:
            child['ended'] = True
        save()
    return 0 if state['status'] == 'stopped' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='Local decoder/projector device; CUDA remains the default')
    parser.add_argument('--hosted-fallback', action='store_true', help='Recover once to existing hosted MoulSot after a local runtime failure')
    parser.add_argument('--max-seconds', type=int, default=0, help='Optional whole supervisor duration; zero runs until stopped')
    args = parser.parse_args()
    if args.max_seconds < 0:
        parser.error('--max-seconds must be nonnegative')
    raise SystemExit(main(args))
