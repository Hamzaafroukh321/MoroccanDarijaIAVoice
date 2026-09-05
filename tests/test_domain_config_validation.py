"""Validate merged preview identity before malformed fields reach task state."""

import json
from copy import deepcopy

import pytest

from engine.config import ROOT, load_config
import engine.demo as demo_module
from engine.scoped_task import ScopedTaskState


def local_profile(monkeypatch, tmp_path, *, renamed_doctor=None, change=None):
    config = load_config(ROOT / 'configs/clinic.json')
    profile = json.loads((ROOT / 'configs/demo/clinic.json').read_text(encoding='utf-8'))
    shared = (ROOT / 'configs/demo/voice.json').read_bytes()
    if renamed_doctor is not None:
        profile['slot_overrides']['doctor']['id'] = renamed_doctor
    if change is not None:
        change(profile)
    directory = tmp_path / 'configs/demo'
    directory.mkdir(parents=True)
    (directory / 'clinic.json').write_text(json.dumps(profile), encoding='utf-8')
    (directory / 'voice.json').write_bytes(shared)
    monkeypatch.setattr(demo_module, 'ROOT', tmp_path)
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'groq')
    return config


@pytest.mark.parametrize('renamed_doctor', ['date', 'renamed_doctor'], ids=['duplicate_identity', 'unknown_identity'])
def test_profile_override_cannot_rename_required_slot(monkeypatch, tmp_path, renamed_doctor):
    config = local_profile(monkeypatch, tmp_path, renamed_doctor=renamed_doctor)
    # Previously doctor->date passed profile loading and dictionary construction
    # silently discarded doctor; date/time alone then became confirmable.
    with pytest.raises(ValueError, match='(?i)slot|override|identity'):
        demo_module.demo_config(config)


def test_unmodified_profile_preserves_required_doctor(monkeypatch, tmp_path):
    config = demo_module.demo_config(local_profile(monkeypatch, tmp_path))
    assert [slot['id'] for slot in config['slots']] == ['doctor', 'date', 'time']
    state = ScopedTaskState(config)
    state.apply([{'op': 'set', 'slot': 'date', 'value': '2026-09-15'},
                 {'op': 'set', 'slot': 'time', 'value': '10:30'}])
    state.begin_confirmation(state.version)
    action = state.consume({'intent': 'task', 'ops': [], 'unclear': False,
                            'is_affirmation': True, 'is_negation': False})
    assert not state.ready and not state.confirmed
    assert action.kind == 'ask' and action.slot == 'doctor'


@pytest.mark.parametrize('target', ['unknown_field', 'patient_name'], ids=['unknown', 'excluded_from_preview'])
def test_override_target_must_be_a_selected_slot(monkeypatch, tmp_path, target):
    def change(profile):
        profile['slot_overrides'][target] = {'required': True}

    config = local_profile(monkeypatch, tmp_path, change=change)
    with pytest.raises(ValueError, match='(?i)slot|override|field'):
        demo_module.demo_config(config)


@pytest.mark.parametrize('invalid_mapping', ['container', 'entry'])
def test_overrides_require_mappings(monkeypatch, tmp_path, invalid_mapping):
    def change(profile):
        if invalid_mapping == 'container':
            profile['slot_overrides'] = []
        else:
            profile['slot_overrides']['doctor'] = []

    config = local_profile(monkeypatch, tmp_path, change=change)
    with pytest.raises(ValueError, match='(?i)slot|override|mapping'):
        demo_module.demo_config(config)


def test_identity_preserving_override_is_allowed(monkeypatch, tmp_path):
    config = demo_module.demo_config(local_profile(monkeypatch, tmp_path, renamed_doctor='doctor'))
    slots = {slot['id']: slot for slot in config['slots']}
    assert set(slots) == {'doctor', 'date', 'time'}
    assert slots['doctor']['values'] == ['doctor_a', 'doctor_b']
    assert slots['doctor']['required'] is True


@pytest.mark.parametrize('domain,expected', [
    ('pizza', {'quantity', 'size', 'toppings', 'drink'}),
    ('clinic', {'doctor', 'date', 'time'}),
])
def test_current_domain_profiles_still_load(monkeypatch, domain, expected):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'groq')
    config = demo_module.demo_config(load_config(ROOT / 'configs' / f'{domain}.json'))
    assert {slot['id'] for slot in config['slots']} == expected
    assert len(config['slots']) == len(expected)


@pytest.mark.parametrize('fields', [
    ['doctor', 'date', 'doctor'], 'doctor', {'doctor': True}, [],
], ids=['duplicate_fields', 'string_fields', 'mapping_fields', 'empty_fields'])
def test_preview_fields_require_a_nonempty_unique_list(monkeypatch, tmp_path, fields):
    config = local_profile(monkeypatch, tmp_path, change=lambda profile: profile.update(fields=fields))
    with pytest.raises(ValueError, match='(?i)fields|slot'):
        demo_module.demo_config(config)


def test_each_selected_source_slot_must_exist_once(monkeypatch, tmp_path):
    config = local_profile(monkeypatch, tmp_path)
    config['slots'].append(deepcopy(next(slot for slot in config['slots'] if slot['id'] == 'doctor')))
    with pytest.raises(ValueError, match='(?i)exactly once|duplicate|slot'):
        demo_module.demo_config(config)


@pytest.mark.parametrize('context', [
    {'enabled': True, 'terms': ['a' * 80, 'b' * 80]},
    {'enabled': False, 'terms': ['hidden\u200bterm']},
    {'enabled': 'yes', 'terms': []},
    {'enabled': True, 'terms': ['term'], 'prompt': 'extra field'},
])
def test_context_structure_and_semantic_bounds_fail_during_domain_loading(tmp_path, context):
    config = json.loads((ROOT/'configs/clinic.json').read_text(encoding='utf-8'))
    config['stt']['moulsot_context'] = context
    path = tmp_path/'clinic.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    with pytest.raises(ValueError):
        load_config(path)


def test_domain_context_remains_local_to_loaded_configuration(tmp_path):
    config = json.loads((ROOT/'configs/clinic.json').read_text(encoding='utf-8'))
    context = {'enabled': True, 'terms': ['الطبيب ألف', 'الطبيب باء']}
    config['stt']['moulsot_context'] = context
    path = tmp_path/'clinic.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    assert load_config(path)['stt']['moulsot_context'] == context
    assert 'moulsot_context' not in load_config(ROOT/'configs/clinic.json')['stt']
    assert 'moulsot_context' not in load_config(ROOT/'configs/pizza.json')['stt']
