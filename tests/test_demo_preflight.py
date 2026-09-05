"""Merged preview prompts must support the task before a voice session starts."""

import json
import sys

import dotenv
import httpx
import pytest

from engine.config import ROOT, load_config
import engine.demo as demo_module
from engine.demo_order import DemoOrderState
from engine.scoped_task import ScopedTaskState
from engine.state import Action
import engine.router as router_module
import engine.stt as stt_module
from scripts import check_config


def malformed_clinic(monkeypatch, tmp_path, case):
    base = load_config(ROOT / 'configs/clinic.json')
    profile = json.loads((ROOT / 'configs/demo/clinic.json').read_text(encoding='utf-8'))
    shared = (ROOT / 'configs/demo/voice.json').read_bytes()
    if case == 'missing_question':
        del profile['questions']['doctor']
    elif case == 'omitted_readback':
        profile['readback_order'].remove('doctor')
    elif case == 'wrong_format':
        profile['readback_formats']['doctor'] = 'day_month_year'
    else:
        raise AssertionError(case)
    directory = tmp_path / 'configs/demo'
    directory.mkdir(parents=True)
    (directory / 'clinic.json').write_text(json.dumps(profile), encoding='utf-8')
    (directory / 'voice.json').write_bytes(shared)
    monkeypatch.setattr(demo_module, 'ROOT', tmp_path)
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'groq')
    return base


@pytest.mark.parametrize('case,diagnostic', [
    ('missing_question', '(?i)question|doctor'),
    ('omitted_readback', '(?i)readback|doctor'),
    ('wrong_format', '(?i)format|doctor|date'),
])
def test_malformed_voice_mapping_fails_at_profile_loading(monkeypatch, tmp_path, case, diagnostic):
    config = malformed_clinic(monkeypatch, tmp_path, case)
    # These previously loaded, then either crashed while rendering or allowed
    # confirmation of a required doctor that the readback never mentioned.
    with pytest.raises(ValueError, match=diagnostic):
        demo_module.demo_config(config)


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_current_profiles_ask_and_read_back_every_required_field(monkeypatch, domain):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'groq')
    config = demo_module.demo_config(load_config(ROOT / 'configs' / f'{domain}.json'))
    if domain == 'clinic':
        state = ScopedTaskState(config)
        values = {'doctor': 'doctor_a', 'date': '2026-09-15', 'time': '10:30'}
        operations = [{'op': 'set', 'slot': key, 'value': value} for key, value in values.items()]
    else:
        state = DemoOrderState(config)
        operations = [{'op': 'create', 'item_id': 1, 'slot': None, 'value': None}]
        for slot, value in {'quantity': 1, 'size': 'large', 'toppings': ['cheese'], 'drink': ['water']}.items():
            operations.append({'op': 'set', 'item_id': None if slot == 'drink' else 1,
                               'slot': slot, 'value': value})
    action = state.next_action()
    question = demo_module.reply_text(action, state.values, config['demo'])
    assert config['demo']['questions'][action.slot] in question
    state.apply(operations)
    assert state.ready and not state.confirmed
    text = demo_module.reply_text(Action('readback'), state.values, config['demo'])
    for slot in config['slots']:
        if slot['required']:
            assert config['demo']['labels'][slot['id']] in text
    assert config['demo']['readback_question'] in text
    if domain == 'clinic':
        assert config['demo']['values']['doctor_a'] in text
        assert '15 / 09 / 2026' in text and '10 : 30' in text
    else:
        assert all(config['demo']['values'][value] in text for value in ('large', 'cheese', 'water'))
    state.begin_confirmation(state.version)
    accepted = state.consume({'intent': 'task', 'ops': [], 'unclear': False,
                              'is_affirmation': True, 'is_negation': False})
    assert accepted.kind == 'accepted' and state.confirmed


def test_missing_pizza_question_cannot_remove_an_adapter_required_field(monkeypatch, tmp_path):
    base = load_config(ROOT / 'configs/pizza.json')
    profile = json.loads((ROOT / 'configs/demo/voice.json').read_text(encoding='utf-8'))
    del profile['questions']['size']
    directory = tmp_path / 'configs/demo'
    directory.mkdir(parents=True)
    (directory / 'voice.json').write_text(json.dumps(profile), encoding='utf-8')
    monkeypatch.setattr(demo_module, 'ROOT', tmp_path)
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'groq')
    with pytest.raises(ValueError, match='(?i)size|question|field'):
        demo_module.demo_config(base)


def test_optional_selected_field_remains_optional_but_is_read_back_when_supplied(monkeypatch, tmp_path):
    base = load_config(ROOT / 'configs/clinic.json')
    profile = json.loads((ROOT / 'configs/demo/clinic.json').read_text(encoding='utf-8'))
    profile['required_slots'].remove('time')
    profile['slot_overrides']['time'] = {'required': False}
    directory = tmp_path / 'configs/demo'
    directory.mkdir(parents=True)
    (directory / 'clinic.json').write_text(json.dumps(profile), encoding='utf-8')
    (directory / 'voice.json').write_bytes((ROOT / 'configs/demo/voice.json').read_bytes())
    monkeypatch.setattr(demo_module, 'ROOT', tmp_path)
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'groq')

    config = demo_module.demo_config(base)
    state = ScopedTaskState(config)
    state.apply([{'op': 'set', 'slot': 'doctor', 'value': 'doctor_a'},
                 {'op': 'set', 'slot': 'date', 'value': '2026-09-15'}])
    assert state.ready and state.next_action().kind == 'readback'
    text = demo_module.reply_text(Action('readback'), state.values, config['demo'])
    assert config['demo']['labels']['time'] not in text
    state.apply([{'op': 'set', 'slot': 'time', 'value': '10:30'}])
    assert state.ready
    text = demo_module.reply_text(Action('readback'), state.values, config['demo'])
    assert config['demo']['labels']['time'] in text and '10 : 30' in text


def offline_cli(monkeypatch, path, *flags):
    def forbidden(*args, **kwargs):
        raise AssertionError('Offline config validation must not allocate a provider/client.')

    monkeypatch.setattr(httpx, 'AsyncClient', forbidden)
    monkeypatch.setattr(httpx, 'Client', forbidden)
    monkeypatch.setattr(demo_module, 'DemoVoice', forbidden)
    monkeypatch.setattr(router_module, 'Router', forbidden)
    monkeypatch.setattr(stt_module, 'SpeechToText', forbidden)
    # Do not read credentials in tests; the CLI's environment-loading hook is
    # independent of the offline profile validation under test.
    monkeypatch.setattr(dotenv, 'load_dotenv', lambda *args, **kwargs: False)
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'groq')
    monkeypatch.setattr(sys, 'argv', ['check_config.py', str(path), *flags])
    return check_config.main()


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_demo_cli_validates_current_profiles_without_provider_allocation(monkeypatch, capsys, domain):
    assert offline_cli(monkeypatch, ROOT / 'configs' / f'{domain}.json', '--demo') == 0
    output = capsys.readouterr()
    assert 'VALID DEMO:' in output.out
    assert 'Offline configuration only' in output.out
    assert 'not verified' in output.out
    assert not output.err


@pytest.mark.parametrize('case', ['missing_question', 'omitted_readback', 'wrong_format'])
def test_demo_cli_reports_malformed_profile_before_provider_allocation(monkeypatch, tmp_path, capsys, case):
    base = malformed_clinic(monkeypatch, tmp_path, case)
    path = tmp_path / 'clinic.json'
    path.write_text(json.dumps(base), encoding='utf-8')
    assert offline_cli(monkeypatch, path, '--demo') == 2
    output = capsys.readouterr()
    assert 'INVALID DEMO:' in output.err
    assert any(word in output.err.lower() for word in ('doctor', 'question', 'readback', 'format'))
    assert 'VALID DEMO:' not in output.out


def test_ready_cli_keeps_research_bank_review_gate(monkeypatch, capsys):
    assert offline_cli(monkeypatch, ROOT / 'configs/clinic.json', '--ready') == 2
    output = capsys.readouterr()
    assert 'NOT READY:' in output.err and 'Language review' in output.err
    assert 'VALID DEMO:' not in output.out


def test_demo_and_research_ready_flags_are_mutually_exclusive(monkeypatch, capsys):
    with pytest.raises(SystemExit) as rejected:
        offline_cli(monkeypatch, ROOT / 'configs/clinic.json', '--demo', '--ready')
    assert rejected.value.code == 2
    assert 'not allowed with argument' in capsys.readouterr().err
