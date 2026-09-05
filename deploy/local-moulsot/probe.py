"""One bounded cold MoulSot transcription; isolated from the running demo."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from prepare import DEST, ROOT


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
        (name, ctypes.c_ulonglong) for name in
        ('total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'extended')]


def available_ram_mib():
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError()
    return memory.avail_phys / 2**20


def free_gpu_mib():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=5,
                            creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    return int(result.stdout.strip().splitlines()[0])


def run(args):
    executables = list((DEST / 'runtime').rglob('llama-mtmd-cli.exe'))
    if len(executables) != 1:
        raise ValueError('Prepare exactly one verified portable runtime first')
    if not (DEST / 'verified-assets.json').is_file():
        raise ValueError('Verified asset manifest is required')
    audio = Path(args.audio).resolve(strict=True)
    env = os.environ.copy()
    dll_dirs = {str(path.parent) for path in (DEST / 'runtime').rglob('*.dll')}
    env['PATH'] = os.pathsep.join(sorted(dll_dirs)) + os.pathsep + env.get('PATH', '')
    command = [str(executables[0]), '-m', str(DEST / 'models/moulsot.v0.3.Q4_K_M.gguf'),
               '--mmproj', str(DEST / 'models/moulsot.v0.3.mmproj-Q8_0.gguf'),
               '--audio', str(audio), '-p', 'a', '-n', '128', '-c', '2048', '--temp', '0',
               '-ngl', '99', '--no-warmup', '--no-mmproj-offload', '-t', '4', '-b', '128', '-ub', '64']
    report = {'purpose': 'One local quantized MoulSot cold compatibility diagnostic, not native accuracy',
              'command': command, 'audio_sha256': hashlib.sha256(audio.read_bytes()).hexdigest(),
              'deadline_seconds': args.deadline, 'samples': [], 'fallback': False, 'production_changed': False}
    prefix = ROOT / 'bench/results' / time.strftime('local_moulsot_cold_%Y%m%d_%H%M%S')
    before = {'free_ram_mib': available_ram_mib(), 'free_gpu_mib': free_gpu_mib()}
    report['before'] = before
    if before['free_ram_mib'] < 2300 or before['free_gpu_mib'] < 1700:
        report['status'] = 'not_started_resource_headroom'
    else:
        started = time.perf_counter()
        with prefix.with_suffix('.stdout.log').open('wb') as stdout, prefix.with_suffix('.stderr.log').open('wb') as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            report['pid'] = process.pid
            report['status'] = 'finished'
            try:
                while process.poll() is None:
                    elapsed = time.perf_counter() - started
                    sample = {'elapsed_seconds': elapsed, 'free_ram_mib': available_ram_mib(), 'free_gpu_mib': free_gpu_mib()}
                    report['samples'].append(sample)
                    if elapsed > args.deadline or sample['free_ram_mib'] < 768 or sample['free_gpu_mib'] < 256:
                        report['status'] = 'stopped_at_resource_or_time_bound'
                        process.kill()
                        break
                    time.sleep(1)
            finally:
                if process.poll() is None:
                    process.kill()
                report['exit_code'] = process.wait(timeout=10)
            report['cold_process_elapsed_ms'] = (time.perf_counter() - started) * 1000
        report['stdout'] = str(prefix.with_suffix('.stdout.log'))
        report['stderr'] = str(prefix.with_suffix('.stderr.log'))
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(prefix.with_suffix('.json')), 'status': report['status'], 'exit_code': report.get('exit_code')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--deadline', type=float, default=120)
    run(parser.parse_args())
