import asyncio
from copy import deepcopy

import pytest

from bench.prepare_speech import fixed_replies, prepare_speech
from engine.config import ROOT, load_config
from engine.demo import chunks, demo_config, reply_text
from engine.state import Action
from engine.stt import wav_bytes


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    result = demo_config(load_config(ROOT / 'configs/clinic.json'))
    # Keep budget tests legible while retaining the real runtime rendering path.
    result['slots'] = [{'id': 'doctor', 'required': True, 'values': ['doctor_a']}]
    result['demo']['questions'] = {'doctor': 'which doctor'}
    result['demo']['menu_options'] = {'doctor': ['doctor A']}
    result['demo']['responses'] = {'greeting': 'hello', 'listen': 'listen',
        'accepted': 'okay', 'handoff': 'bye', 'unsupported_option': 'choose'}
    result['demo']['tts_max_chars'] = 100
    return result


def factory_for(config, root, *, fail=False):
    calls = []
    instances = []

    class Voice:
        def __init__(self, actual_config, actual_root):
            assert actual_config is config
            assert actual_root == root.resolve()
            self.closed = False
            instances.append(self)

        async def render(self, action, values):
            from engine.demo import read_speech_cache, speech_cache_path
            text = reply_text(action, values, config['demo'])
            for part in chunks(text, config['demo']['tts_max_chars']):
                path = speech_cache_path(config['demo'], root, part)
                if read_speech_cache(path, config['runtime']) is None:
                    calls.append(part)
                    if fail:
                        raise RuntimeError('private provider response must not appear in report')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(wav_bytes(b'\x01\x00' * 160, config['runtime']))

        async def close(self):
            self.closed = True

    return Voice, calls, instances


def run(config, root, **kwargs):
    return asyncio.run(prepare_speech(config, root, **kwargs))


def test_dry_run_never_constructs_producer_or_writes_cache(config, tmp_path):
    factory, calls, instances = factory_for(config, tmp_path)
    report = run(config, tmp_path, voice_factory=factory)
    assert report['mode'] == 'dry_run'
    assert not calls and not instances and not list(tmp_path.iterdir())
    assert report['available_texts'] == []
    assert report['reserved_misses'] == 3
    assert any(r['status'] == 'skipped_budget' for r in report['replies'])


@pytest.mark.parametrize('kwargs', [
    {'max_misses': -1}, {'max_misses': 21}, {'max_misses': True},
    {'max_characters': -1}, {'max_characters': 3001}, {'max_characters': 3.5},
])
def test_invalid_bounds_rejected_before_producer(config, tmp_path, kwargs):
    factory, calls, instances = factory_for(config, tmp_path)
    with pytest.raises(ValueError):
        run(config, tmp_path, execute=True, voice_factory=factory, **kwargs)
    assert not calls and not instances


@pytest.mark.parametrize('kwargs', [{'max_misses': 0}, {'max_characters': 0}])
def test_zero_budget_never_generates(config, tmp_path, kwargs):
    factory, calls, instances = factory_for(config, tmp_path)
    report = run(config, tmp_path, execute=True, voice_factory=factory, **kwargs)
    assert not calls and not instances
    assert all(r['status'] == 'skipped_budget' for r in report['replies'])


def test_only_whole_reply_fitting_both_limits_is_generated(config, tmp_path):
    config['demo']['responses']['greeting'] = 'hello world'
    config['demo']['tts_max_chars'] = 6
    factory, calls, instances = factory_for(config, tmp_path)
    report = run(config, tmp_path, execute=True, max_misses=1,
                 max_characters=5, voice_factory=factory)
    first = report['replies'][0]
    assert first['missing_fragments'] == 2
    assert first['status'] == 'skipped_budget'
    assert 'hello' not in calls and 'world' not in calls
    assert report['reserved_misses'] <= 1
    assert report['reserved_characters'] <= 5


def test_exact_boundary_and_duplicate_text_skipped(config, tmp_path):
    config['demo']['responses']['listen'] = 'hello'
    factory, calls, instances = factory_for(config, tmp_path)
    report = run(config, tmp_path, execute=True, max_misses=1,
                 max_characters=5, voice_factory=factory)
    assert calls == ['hello']
    assert instances[0].closed
    assert report['reserved_characters'] == 5
    matching = [r for r in report['replies'] if r['text'] == 'hello']
    assert len(matching) == 1
    assert {a['kind'] for a in matching[0]['actions']} == {'greeting', 'listen'}
    assert matching[0]['available']


def test_valid_cache_skips_and_corrupt_cache_is_only_replaced_on_execute(config, tmp_path):
    from engine.demo import speech_cache_path
    valid = speech_cache_path(config['demo'], tmp_path, 'hello')
    corrupt = speech_cache_path(config['demo'], tmp_path, 'which doctor')
    valid.parent.mkdir(parents=True)
    original = wav_bytes(b'\x01\x00' * 160, config['runtime'])
    valid.write_bytes(original)
    corrupt.write_bytes(b'broken WAV')
    factory, calls, instances = factory_for(config, tmp_path)
    dry = run(config, tmp_path, voice_factory=factory)
    assert dry['replies'][0]['cache_hit']
    assert dry['replies'][1]['fragments_before'][0]['cache'] == 'corrupt'
    assert corrupt.read_bytes() == b'broken WAV'
    report = run(config, tmp_path, execute=True, max_misses=1, voice_factory=factory)
    assert calls == ['which doctor']
    assert valid.read_bytes() == original
    assert report['replies'][1]['status'] == 'prepared'
    assert 'which doctor' in report['available_texts']


def test_failure_stops_generation_and_does_not_export_provider_details(config, tmp_path):
    factory, calls, instances = factory_for(config, tmp_path, fail=True)
    report = run(config, tmp_path, execute=True, voice_factory=factory)
    assert calls == ['hello']
    assert report['stopped_on_failure'] and instances[0].closed
    assert report['replies'][0]['error_type'] == 'RuntimeError'
    assert all(r['status'] == 'skipped_after_failure' for r in report['replies'][1:])
    assert 'private provider' not in str(report)


def test_shared_fragments_reserved_once(config, tmp_path):
    config['demo']['tts_max_chars'] = 6
    config['demo']['responses']['greeting'] = 'hello world'
    config['demo']['questions']['doctor'] = 'hello there'
    factory, calls, _ = factory_for(config, tmp_path)
    dry = run(config, tmp_path, max_misses=3, max_characters=15, voice_factory=factory)
    assert dry['replies'][0]['status'] == dry['replies'][1]['status'] == 'would_prepare'
    actual = run(config, tmp_path, execute=True, max_misses=3,
                 max_characters=15, voice_factory=factory)
    assert calls == ['hello', 'world', 'there']
    assert actual['reserved_misses'] == dry['reserved_misses'] == 3
    assert actual['reserved_characters'] == dry['reserved_characters'] == 15


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_real_profiles_enumerate_fixed_runtime_text_only(domain, monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    config = demo_config(load_config(ROOT / 'configs' / f'{domain}.json'))
    original = deepcopy(config)
    replies = fixed_replies(config)
    assert len({r['text'] for r in replies}) == len(replies)
    for reply in replies:
        for provenance in reply['actions']:
            action = Action(**provenance)
            assert action.kind != 'readback'
            assert action.item_id in (None, 1)
            assert reply_text(action, reply['values'], config['demo']) == reply['text']
    assert config == original


def test_cli_defaults_to_dry_run(monkeypatch, tmp_path, capsys):
    import json
    import bench.prepare_speech as cli
    seen = {}

    async def fake_prepare(config, root, **kwargs):
        seen.update(kwargs)
        return {'mode': 'dry_run', 'available_texts': [], 'reserved_misses': 0,
                'reserved_characters': 0, 'stopped_on_failure': False}

    monkeypatch.setattr(cli, 'prepare_speech', fake_prepare)
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    output = tmp_path / 'plan.json'
    assert cli.main(['--domain', 'clinic', '--output', str(output)]) == 0
    assert seen == {'execute': False, 'max_misses': 3, 'max_characters': 500}
    assert json.loads(output.read_text(encoding='utf-8'))['mode'] == 'dry_run'
    assert json.loads(capsys.readouterr().out)['mode'] == 'dry_run'
