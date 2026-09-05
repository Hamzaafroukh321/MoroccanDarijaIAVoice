"""Supervisor orchestration only: fake jobs, clock and HTTP; no child services."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import pytest


@pytest.fixture
def rig(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / 'deploy/local-moulsot/run_local_demo.py'
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location('supervisor_under_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dest = tmp_path / '.local/moulsot'
    (dest / 'runtime').mkdir(parents=True)
    (dest / 'runtime/llama-server.exe').write_bytes(b'fake')
    (tmp_path / 'bench/results').mkdir(parents=True)
    env_file = tmp_path / '.env'
    env_file.write_text('UNCHANGED=fixture\n', encoding='utf-8')
    for name, value in {'ROOT': tmp_path, 'DEST': dest, 'STATE': dest / 'supervisor.json',
                        'STOP': dest / 'STOP'}.items():
        monkeypatch.setattr(module, name, value)
    real_preflight = module.preflight
    monkeypatch.setattr(module, 'preflight', lambda *args, **kwargs: {'ready': True, 'issues': []})
    monkeypatch.setattr(module, 'available_ram_mib', lambda: 9999)
    monkeypatch.setattr(module, 'free_gpu_mib', lambda: 9999)
    clock = SimpleNamespace(now=0.0)
    def sleep(seconds):
        clock.now += seconds
    monkeypatch.setattr(module, 'time', SimpleNamespace(perf_counter=lambda: clock.now,
        sleep=sleep, strftime=lambda *_: 'fixture'))
    control = SimpleNamespace(jobs=[], calls=[], requests=[], ready_count=0,
        model_id='moulsot.v0.3.Q4_K_M.gguf', issues=[], stop_after_ready=True,
        fail_local=False, fail_hosted=False, never_ready=False, spawn_error=None)

    class Job:
        def __init__(self):
            self.closed = 0
            self.index = len(control.jobs)
            control.jobs.append(self)
        def spawn(self, command, **kwargs):
            name = ('model' if command[0].endswith('llama-server.exe') else
                    'bridge' if 'bridge:app' in command else 'engine')
            control.calls.append((name, list(command), kwargs))
            if control.spawn_error == name:
                raise RuntimeError('simulated spawn failure')
            def poll():
                if control.ready_count >= 2 and self.index == 0 and control.fail_local:
                    return 1
                if control.ready_count >= 3 and self.index == 1 and control.fail_hosted:
                    return 1
                return None
            return SimpleNamespace(pid=100 + len(control.calls), poll=poll)
        def close(self):
            self.closed += 1
    monkeypatch.setitem(sys.modules, 'windows_job', SimpleNamespace(WindowsJob=Job))

    class Client:
        def __init__(self, **kwargs):
            assert kwargs.get('trust_env') is False
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            control.requests.append(url)
            request = httpx.Request('GET', url)
            if control.never_ready:
                return httpx.Response(503, request=request)
            if url.endswith('/v1/models'):
                return httpx.Response(200, json={'data': [{'id': control.model_id}]}, request=request)
            if '8000/' in url:
                control.ready_count += 1
                domain = 'clinic' if 'domain=clinic' in url else 'pizza'
                payload = {'domain_id': domain, 'initial_mode': 'demo', 'demo_supported': True,
                    'demo_issues': control.issues, 'demo_tts_provider': 'darija_xtts'}
                if control.stop_after_ready and control.ready_count == 2:
                    module.STOP.write_text('stop', encoding='utf-8')
                return httpx.Response(200, text='<script id="capture-config" type="application/json">'
                    + json.dumps(payload) + '</script>', request=request)
            return httpx.Response(200, json={'status': 'ok', 'adapter': 'local-moulsot'}, request=request)
    monkeypatch.setattr(module.httpx, 'Client', Client)
    monkeypatch.setenv('GROQ_API_KEY', 'secret-fixture-never-serialize')
    monkeypatch.setenv('MOULSOT_ENDPOINT', 'https://example.hf.space')
    monkeypatch.setenv('MOULSOT_PROTOCOL', 'gradio')
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    args = SimpleNamespace(start=True, hosted_fallback=False, max_seconds=8)
    def run():
        result = module.main(args)
        assert env_file.read_text(encoding='utf-8') == 'UNCHANGED=fixture\n'
        for path in tmp_path.rglob('*.json'):
            assert 'secret-fixture-never-serialize' not in path.read_text(encoding='utf-8')
        return result
    return SimpleNamespace(module=module, control=control, args=args, clock=clock, run=run,
        real_preflight=real_preflight, state=lambda: json.loads(module.STATE.read_text(encoding='utf-8')))


def test_order_correct_engine_entrypoint_and_stop_closes_job(rig):
    assert rig.run() == 0
    calls = rig.control.calls
    assert [call[0] for call in calls] == ['model', 'bridge', 'engine']
    model_command = calls[0][1]
    assert int(model_command[model_command.index('-n') + 1]) == rig.module.MAX_OUTPUT_TOKENS == 512
    assert calls[2][1][-2:] == ['-m', 'engine.server']
    assert 'uvicorn' not in calls[2][1]
    assert calls[2][2]['env']['MOULSOT_PROTOCOL'] == 'json'
    assert calls[2][2]['env']['MOULSOT_ENDPOINT'] == 'http://127.0.0.1:8012/transcribe'
    assert calls[2][2]['env']['STT_PRIMARY'] == 'moulsot'
    assert calls[2][2]['env']['DEMO_TTS_PROVIDER'] == 'darija_xtts'
    assert all(job.closed == 1 for job in rig.control.jobs)
    assert rig.state()['status'] == 'stopped'


def test_wrong_model_never_starts_bridge(rig):
    rig.control.model_id = 'unrelated-model.gguf'
    assert rig.run() == 1
    assert [call[0] for call in rig.control.calls] == ['model']
    assert rig.control.jobs[0].closed == 1
    assert 'identity' in rig.state()['error'].lower()


def test_http_200_with_demo_issues_is_not_ready(rig):
    rig.control.issues = ['Missing speech configuration']
    assert rig.run() == 1
    assert rig.state()['status'] == 'failed'
    assert all(job.closed == 1 for job in rig.control.jobs)


def test_spawn_failure_closes_owned_tree(rig):
    rig.control.spawn_error = 'bridge'
    assert rig.run() == 1
    assert [call[0] for call in rig.control.calls] == ['model', 'bridge']
    assert rig.control.jobs[0].closed == 1


def test_hosted_recovery_once_preserves_inherited_configuration(rig):
    rig.args.hosted_fallback = True
    rig.control.stop_after_ready = False
    rig.control.fail_local = rig.control.fail_hosted = True
    assert rig.run() == 1
    assert [call[0] for call in rig.control.calls] == ['model', 'bridge', 'engine', 'engine']
    env = rig.control.calls[-1][2]['env']
    assert env['MOULSOT_ENDPOINT'] == 'https://example.hf.space'
    assert env['MOULSOT_PROTOCOL'] == 'gradio'
    assert env['STT_PRIMARY'] == 'moulsot'
    assert env['DEMO_TTS_PROVIDER'] == 'darija_xtts'
    assert len(rig.control.jobs) == 2
    assert all(job.closed == 1 for job in rig.control.jobs)
    assert rig.state()['recovered_once'] is True


def test_stop_marker_during_startup_closes_without_more_services(rig):
    rig.module.STOP.write_text('stop', encoding='utf-8')
    # Real preflight rejects this earlier; exercise the startup race after preflight.
    assert rig.run() == 0
    assert len(rig.control.calls) <= 1
    assert all(job.closed == 1 for job in rig.control.jobs)


def test_total_duration_includes_startup(rig):
    rig.control.never_ready = True
    rig.args.max_seconds = 2
    rig.run()
    assert rig.clock.now <= 2.5
    assert [call[0] for call in rig.control.calls] == ['model']
    assert all(job.closed == 1 for job in rig.control.jobs)


def test_dry_preflight_never_allocates_job(rig):
    rig.args.start = False
    assert rig.run() == 0
    assert not rig.control.jobs
    assert not rig.control.requests


def test_retired_local_logs_do_not_stop_successful_hosted_recovery(rig, monkeypatch):
    rig.args.hosted_fallback = True
    rig.control.stop_after_ready = False
    retired_log = rig.module.ROOT / 'bench/results/fixture.model.stdout.log'
    retired_log.write_text('small fixture standing in for a large retired log', encoding='utf-8')
    original_stat = Path.stat
    def stat(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        if path == retired_log:
            fields = list(result)
            fields[6] = 64 * 1024 * 1024 + 1
            return type(result)(fields)
        return result
    monkeypatch.setattr(Path, 'stat', stat)

    assert rig.run() == 0
    assert [call[0] for call in rig.control.calls] == ['model', 'bridge', 'engine', 'engine']
    state = rig.state()
    assert state['mode'] == 'hosted'
    assert state['status'] == 'stopped'
    assert state['recovered_once'] is True
    assert state['failure'] == 'Supervisor log limit reached (64 MiB)'
    assert state['log_bytes'] == 0
    # Hosted monitoring survives several cycles until the configured duration.
    assert rig.clock.now >= rig.args.max_seconds
    assert len(rig.control.jobs) == 2
    assert all(job.closed == 1 for job in rig.control.jobs)
    assert retired_log.exists()


def test_cpu_supervision_uses_explicit_cpu_flags_and_never_queries_gpu(rig, monkeypatch):
    rig.args.device = 'cpu'
    rig.control.stop_after_ready = False
    def forbidden():
        pytest.fail('CPU supervision must not query GPU memory')
    monkeypatch.setattr(rig.module, 'free_gpu_mib', forbidden)
    assert rig.run() == 0
    command = rig.control.calls[0][1]
    assert command[command.index('--device') + 1] == 'none'
    assert command[command.index('-ngl') + 1] == '0'
    assert '--no-mmproj-offload' in command
    assert command[command.index('-n') + 1] == str(rig.module.MAX_OUTPUT_TOKENS)
    assert rig.state()['selected_device'] == 'cpu'
    assert rig.clock.now >= rig.args.max_seconds
    assert rig.control.jobs[0].closed == 1


def test_default_cuda_flags_and_gpu_guards_remain_enabled(rig, monkeypatch):
    queried = []
    monkeypatch.setattr(rig.module, 'free_gpu_mib', lambda: queried.append(True) or 9999)
    assert rig.run() == 0
    command = rig.control.calls[0][1]
    assert command[command.index('-ngl') + 1] == '99'
    assert '--no-mmproj-offload' not in command
    assert not ('--device' in command and command[command.index('--device') + 1] == 'none')
    assert rig.state()['selected_device'] == 'cuda'
    assert queried


@pytest.mark.parametrize('device,ram,gpu,ready', [
    ('cpu', 3071, None, False), ('cpu', 3072, None, True),
    ('cuda', 2559, 1900, False), ('cuda', 2560, 1899, False), ('cuda', 2560, 1900, True)])
def test_device_preflight_memory_thresholds(rig, monkeypatch, device, ram, gpu, ready):
    configs = rig.module.ROOT / 'configs'
    configs.mkdir()
    (configs / 'pizza.json').write_text(json.dumps({'runtime': {'host': '127.0.0.1', 'port': 8000}}))
    monkeypatch.setattr(rig.module, 'ASSETS', [])
    monkeypatch.setattr(rig.module, 'occupied_ports', lambda: [])
    monkeypatch.setattr(rig.module, 'available_ram_mib', lambda: ram)
    def gpu_memory():
        assert device == 'cuda', 'CPU preflight queried GPU memory'
        return gpu
    monkeypatch.setattr(rig.module, 'free_gpu_mib', gpu_memory)
    check = rig.real_preflight(device=device)
    assert check['ready'] is ready
    if not ready:
        assert any('headroom' in issue.lower() or 'memory' in issue.lower() for issue in check['issues'])
    assert not rig.control.jobs


@pytest.mark.parametrize('phase', ['startup', 'running'])
def test_cpu_ram_guard_closes_owned_tree_without_gpu_query(rig, monkeypatch, phase):
    rig.args.device = 'cpu'
    rig.control.stop_after_ready = False
    def ram():
        return 767 if phase == 'startup' or rig.control.ready_count >= 2 else 3072
    monkeypatch.setattr(rig.module, 'available_ram_mib', ram)
    monkeypatch.setattr(rig.module, 'free_gpu_mib', lambda: pytest.fail('CPU RAM guard queried GPU'))
    assert rig.run() == 1
    assert rig.state()['status'] == 'failed'
    assert len(rig.control.calls) == (1 if phase == 'startup' else 3)
    assert all(job.closed == 1 for job in rig.control.jobs)


def test_cpu_failure_keeps_one_hosted_fallback_with_original_environment(rig, monkeypatch):
    rig.args.device = 'cpu'
    rig.args.hosted_fallback = True
    rig.control.stop_after_ready = False
    rig.control.fail_local = True
    monkeypatch.setattr(rig.module, 'free_gpu_mib', lambda: pytest.fail('CPU fallback queried GPU'))
    assert rig.run() == 0
    assert len(rig.control.jobs) == 2
    assert [call[0] for call in rig.control.calls] == ['model', 'bridge', 'engine', 'engine']
    hosted = rig.control.calls[-1][2]['env']
    assert hosted['MOULSOT_PROTOCOL'] == 'gradio'
    assert hosted['MOULSOT_ENDPOINT'] == 'https://example.hf.space'
    assert hosted['STT_PRIMARY'] == 'moulsot'
    assert rig.state()['recovered_once'] is True
    assert all(job.closed == 1 for job in rig.control.jobs)
