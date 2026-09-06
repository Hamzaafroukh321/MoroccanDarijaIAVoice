"""Apply optional Space patch only in temporary files; execute stubbed callback."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
from types import SimpleNamespace
from typing import Any

import pytest


DIRECTORY = Path(__file__).resolve().parents[1] / 'deploy/moulsot-space'
SECRET = 'PRIVATE_INPUT_SENTINEL /tmp/private.wav hf_fixture'


class GradioError(Exception):
    pass


@pytest.fixture
def patched(tmp_path):
    originals = {name: (DIRECTORY / name).read_bytes() for name in
        ('app.py', 'app.profiled.py', 'source.json', 'profiling.patch')}
    baseline = originals['app.py']
    metadata = json.loads(originals['source.json'])
    assert hashlib.sha256(baseline).hexdigest() == metadata['patched_sha256']
    destination = tmp_path / 'app.py'
    destination.write_bytes(baseline)
    patch = DIRECTORY / 'error-output.patch'
    for flags in (['--check'], []):
        result = subprocess.run(['git', 'apply', *flags, str(patch)], cwd=tmp_path,
            capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
    yield destination
    assert all((DIRECTORY / name).read_bytes() == content for name, content in originals.items())


def extracted(path, *, parser=None, model=None):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'transcribe')
    calls, parses, decorators = [], [], []
    def predict(**kwargs):
        calls.append(kwargs)
        return model(**kwargs) if model else [SimpleNamespace(text='normal transcript')]
    def parse(value):
        parses.append(value)
        return parser(value) if parser else value
    def gpu(**kwargs):
        decorators.append(kwargs)
        return lambda fn: fn
    namespace = {'Any': Any, 'spaces': SimpleNamespace(GPU=gpu),
        'gr': SimpleNamespace(Progress=lambda **kwargs: None, Error=GradioError),
        'asr': SimpleNamespace(transcribe=predict), '_parse_audio_any': parse,
        'lang_map': {'Arabic display': 'Arabic'}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['transcribe'], calls, parses, decorators


def test_patch_changes_only_three_callback_error_branches_and_leaves_runtime_contract(patched):
    old = ast.parse((DIRECTORY / 'app.py').read_text(encoding='utf-8'))
    new = ast.parse(patched.read_text(encoding='utf-8'))
    old_function = next(node for node in old.body if isinstance(node, ast.FunctionDef) and node.name == 'transcribe')
    new_function = next(node for node in new.body if isinstance(node, ast.FunctionDef) and node.name == 'transcribe')
    assert ast.dump(old_function.args) == ast.dump(new_function.args)
    assert [ast.dump(x) for x in old_function.decorator_list] == [ast.dump(x) for x in new_function.decorator_list]
    assert [ast.dump(node) for node in old.body if node is not old_function] == [
        ast.dump(node) for node in new.body if node is not new_function]
    model_calls = lambda function: [ast.dump(node) for node in ast.walk(function) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
        and node.func.value.id == 'asr']
    assert model_calls(old_function) == model_calls(new_function)
    raises = [node for node in ast.walk(new_function) if isinstance(node, ast.Raise)]
    assert len(raises) == 3
    assert all(isinstance(node.exc, ast.Call) and ast.unparse(node.exc.func) == 'gr.Error'
        and len(node.exc.args) == 1 and isinstance(node.exc.args[0], ast.Constant)
        and isinstance(node.exc.args[0].value, str) for node in raises)
    new_returns = [node for node in ast.walk(new_function) if isinstance(node, ast.Return)]
    assert len(new_returns) == 1
    # Select the original non-tuple success expression regardless of AST walk order.
    original_return = next(node for node in ast.walk(old_function) if isinstance(node, ast.Return)
        and not isinstance(node.value, ast.Tuple))
    assert ast.dump(original_return) == ast.dump(new_returns[0])


@pytest.mark.parametrize('failure', ['missing_upload', 'parse_error', 'result_shape'])
def test_known_failures_raise_static_gradio_errors_at_correct_model_boundary(patched, failure):
    def parser(value):
        if failure == 'parse_error':
            raise ValueError(SECRET)
        return value
    model = (lambda **kwargs: []) if failure == 'result_shape' else None
    function, calls, parses, decorators = extracted(patched, parser=parser, model=model)
    audio = None if failure == 'missing_upload' else ([.1] * 160, 16000)
    with pytest.raises(GradioError) as raised:
        function(audio, 'Auto')
    rendered = ''.join(traceback.format_exception(raised.value))
    assert SECRET not in str(raised.value) and SECRET not in rendered
    assert '<div' not in str(raised.value)
    assert len(calls) == int(failure == 'result_shape')
    assert len(parses) == int(failure != 'missing_upload')
    assert decorators == [{'duration': 60}]


@pytest.mark.parametrize('text,language', [('normal transcript', 'Arabic display'),
    ('', 'Auto'), (None, 'unconfigured language')])
def test_normal_and_empty_success_output_and_model_arguments_remain_unchanged(patched, text, language):
    model = lambda **kwargs: [SimpleNamespace(text=text)]
    original, original_calls, _, _ = extracted(DIRECTORY / 'app.py', model=model)
    updated, calls, _, _ = extracted(patched, model=model)
    audio = ([.25] * 160, 16000)
    assert updated(audio, language) == original(audio, language)
    assert calls == original_calls and len(calls) == 1


@pytest.mark.parametrize('boundary', ['parser_non_value_error', 'model_runtime_error', 'model_type_error'])
def test_unrelated_parser_and_model_exceptions_keep_their_original_identity(patched, boundary):
    error = TypeError(SECRET) if boundary == 'model_type_error' else RuntimeError(SECRET)
    def fail(*args, **kwargs):
        raise error
    parser = fail if boundary == 'parser_non_value_error' else None
    model = fail if boundary != 'parser_non_value_error' else None
    function, calls, parses, _ = extracted(patched, parser=parser, model=model)
    with pytest.raises(type(error)) as raised:
        function(([.1] * 160, 16000), 'Auto')
    assert raised.value is error
    assert len(parses) == 1 and len(calls) == int(boundary != 'parser_non_value_error')
