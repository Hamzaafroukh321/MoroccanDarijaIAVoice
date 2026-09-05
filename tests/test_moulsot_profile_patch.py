"""Execute only extracted profiling helpers/function with fake model and Gradio."""

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import difflib
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest


DIRECTORY = Path(__file__).resolve().parents[1] / 'deploy/moulsot-space'
BASELINE = DIRECTORY / 'app.py'
PROFILED = DIRECTORY / 'app.profiled.py'


def extracted(path, *, enabled=True, parser=None, model=None, printer=None):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and (node.name.startswith('_profile_') or node.name == 'transcribe')]
    records, calls, decorators = [], [], []

    def predict(**kwargs):
        calls.append(kwargs)
        return model(**kwargs) if model else [SimpleNamespace(text='private transcript')]

    def gpu(**kwargs):
        decorators.append(kwargs)
        return lambda fn: fn

    namespace = {
        'Any': Any, 'os': SimpleNamespace(getenv=lambda key: '1' if enabled else None),
        'time': time, 'datetime': datetime, 'timezone': timezone, 'uuid4': uuid4,
        'json': json, 'contextmanager': contextmanager,
        'spaces': SimpleNamespace(GPU=gpu), 'gr': SimpleNamespace(Progress=lambda **kwargs: None),
        'asr': SimpleNamespace(transcribe=predict), 'lang_map': {'Arabic display': 'Arabic'},
        '_parse_audio_any': parser or (lambda value: value),
        'print': printer or (lambda line, **kwargs: records.append(json.loads(line))),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['transcribe'], records, calls, decorators


def test_patch_is_deterministic_and_baseline_contract_is_unchanged():
    original = BASELINE.read_text(encoding='utf-8')
    profiled = PROFILED.read_text(encoding='utf-8')
    digest = hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    assert f'# Baseline app.py SHA256: {digest}' in profiled
    expected = ''.join(difflib.unified_diff(original.splitlines(keepends=True),
        profiled.splitlines(keepends=True), fromfile='a/app.py', tofile='b/app.py'))
    assert (DIRECTORY / 'profiling.patch').read_text(encoding='utf-8') == expected
    old_tree, new_tree = ast.parse(original), ast.parse(profiled)
    old = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'transcribe')
    new = next(node for node in new_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'transcribe')
    assert ast.dump(old.args) == ast.dump(new.args)
    assert [ast.dump(item) for item in old.decorator_list] == [ast.dump(item) for item in new.decorator_list]
    assert sorted(ast.dump(node.value) for node in ast.walk(old) if isinstance(node, ast.Return)) == sorted(
        ast.dump(node.value) for node in ast.walk(new) if isinstance(node, ast.Return))

    def model_configuration(tree):
        return [ast.dump(node) for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == 'from_pretrained']

    assert model_configuration(old_tree) == model_configuration(new_tree)


def test_disabled_profiling_has_no_logs_and_same_model_call():
    original, _, original_calls, _ = extracted(BASELINE)
    profiled, records, calls, decorators = extracted(PROFILED, enabled=False)
    audio = ([.1] * 160, 16000)
    assert profiled(audio, 'Arabic display') == original(audio, 'Arabic display')
    assert calls == original_calls == [{'audio': audio, 'language': 'Arabic'}]
    assert decorators == [{'duration': 60}]
    assert records == []


@pytest.mark.parametrize('clock_matches_sample_digits', [False, True])
def test_success_records_only_safe_metadata_and_host_wall_timings(clock_matches_sample_digits):
    fn, records, calls, _ = extracted(PROFILED)
    if clock_matches_sample_digits:
        # Legitimate elapsed timing may contain the same digits as an audio
        # sample. Privacy is the record schema, not substring coincidence.
        ticks = iter([0, 0, .000025, .000025, .0001, .0001253])
        fn.__globals__['time'] = SimpleNamespace(perf_counter=lambda: next(ticks))
    assert fn(([.125] * 160, 16000), 'Arabic display') == 'private transcript'
    start, end = records
    assert start['event'] == 'moulsot_profile_start' and end['event'] == 'moulsot_profile_end'
    assert start['profile_id'] == end['profile_id']
    assert len(end['profile_id']) == 32
    assert end['outcome'] == 'ok' and end['error_type'] is None
    assert end['samples'] == 160 and end['sample_rate_hz'] == 16000
    assert end['audio_duration_ms'] == 10
    assert end['body_ms'] >= end['parse_ms'] + end['model_call_wall_ms'] >= 0
    assert end['elapsed_clock'] == 'perf_counter'
    assert len(calls) == 1
    assert set(start) == {'event', 'profile_id', 'started_utc'}
    assert set(end) == {'event', 'profile_id', 'started_utc', 'outcome', 'error_type',
        'elapsed_clock', 'samples', 'sample_rate_hz', 'audio_duration_ms', 'body_ms',
        'parse_ms', 'model_call_wall_ms'}
    for record in records:
        assert type(record['profile_id']) is str
        assert len(record['profile_id']) == 32
        assert set(record['profile_id']) <= set('0123456789abcdef')
        assert type(record['started_utc']) is str
        assert datetime.fromisoformat(record['started_utc']).utcoffset().total_seconds() == 0
    assert type(end['samples']) is type(end['sample_rate_hz']) is int
    for field in ('audio_duration_ms', 'body_ms', 'parse_ms', 'model_call_wall_ms'):
        assert type(end[field]) in (int, float)
        assert math.isfinite(end[field]) and end[field] >= 0
    # Exact fields and scalar types exclude input arrays, paths and arbitrary
    # transcript/language strings while allowing all valid numeric timings.
    if clock_matches_sample_digits:
        assert end['body_ms'] == pytest.approx(.1253)


@pytest.mark.parametrize('invalid', ['none', 'parse_error', 'unexpected_result'])
def test_early_returns_preserve_existing_api_values(invalid):
    def parser(value):
        if invalid == 'parse_error':
            raise ValueError('private input detail')
        return value

    model = (lambda **kwargs: []) if invalid == 'unexpected_result' else None
    original, _, _, _ = extracted(BASELINE, parser=parser, model=model)
    profiled, records, _, _ = extracted(PROFILED, parser=parser, model=model)
    audio = None if invalid == 'none' else ([.1] * 160, 16000)
    assert profiled(audio, 'Auto') == original(audio, 'Auto')
    assert records[-1]['outcome'] == ('invalid_result' if invalid == 'unexpected_result' else 'invalid_input')
    assert 'private input detail' not in json.dumps(records)
    if invalid != 'unexpected_result':
        assert records[-1]['model_call_wall_ms'] is None


def test_model_exception_is_preserved_without_logging_its_message():
    failure = RuntimeError('private model message hf_secret /tmp/audio.wav')

    def model(**kwargs):
        raise failure

    fn, records, _, _ = extracted(PROFILED, model=model)
    with pytest.raises(RuntimeError) as raised:
        fn(([.1] * 160, 16000), 'Auto')
    assert raised.value is failure
    assert records[-1]['outcome'] == 'error'
    assert records[-1]['error_type'] == 'RuntimeError'
    assert records[-1]['model_call_wall_ms'] >= 0
    for private in ('private model message', 'hf_secret', '/tmp/audio.wav'):
        assert private not in json.dumps(records)


def test_concurrent_invocations_keep_distinct_profile_records():
    barrier = threading.Barrier(2)

    def model(**kwargs):
        barrier.wait(timeout=2)
        return [SimpleNamespace(text='private transcript')]

    fn, records, _, _ = extracted(PROFILED, model=model)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(fn, ([.1] * 160, 16000), 'Auto')
        second = pool.submit(fn, ([.1] * 320, 16000), 'Auto')
        assert first.result(timeout=3) == second.result(timeout=3) == 'private transcript'
    endings = [record for record in records if record['event'] == 'moulsot_profile_end']
    assert len({record['profile_id'] for record in endings}) == 2
    assert {record['samples'] for record in endings} == {160, 320}
    assert all(sum(record['profile_id'] == event['profile_id'] for event in records) == 2
               for record in endings)


@pytest.mark.parametrize('model_fails', [False, True])
def test_logging_failure_does_not_change_return_or_exception(model_fails):
    failure = RuntimeError('private original failure')

    def model(**kwargs):
        if model_fails:
            raise failure
        return [SimpleNamespace(text='private transcript')]

    def broken_printer(*args, **kwargs):
        raise OSError('offline broken logging sink')

    fn, records, calls, _ = extracted(PROFILED, model=model, printer=broken_printer)
    if model_fails:
        with pytest.raises(RuntimeError) as raised:
            fn(([.1] * 160, 16000), 'Auto')
        assert raised.value is failure
    else:
        assert fn(([.1] * 160, 16000), 'Auto') == 'private transcript'
    assert records == [] and len(calls) == 1
