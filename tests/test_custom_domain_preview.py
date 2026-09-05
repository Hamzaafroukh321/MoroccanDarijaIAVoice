"""An English diagnostic third domain exercises configurable flat previews."""

import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from engine.config import ROOT, load_config
import engine.demo as demo_module
import engine.server as server_module
from engine.router import build_messages, parse_response, response_format
from engine.scoped_task import ScopedTaskState
from engine.state import Action


@pytest.fixture
def third_domain(monkeypatch, tmp_path):
    base = json.loads((ROOT / 'configs/clinic.json').read_text(encoding='utf-8').replace('doctor', 'counter'))
    base.update(domain_id='service-desk', display_name='Counter service diagnostic')
    profile = json.loads((ROOT / 'configs/demo/clinic.json').read_text(encoding='utf-8').replace('doctor', 'counter'))
    profile.update(label='Counter preference diagnostic',
                   task_scope='fictional service counter, date and time preferences; no external action',
                   language_review='English diagnostic fixture only; not native language evaluation',
                   questions={'counter': 'Choose counter A or B.', 'date': 'Which date?', 'time': 'Which time?'},
                   labels={'counter': 'Counter', 'date': 'Date', 'time': 'Time'},
                   values={'counter_a': 'Counter A', 'counter_b': 'Counter B'},
                   readback_prefix='Your diagnostic preferences:', readback_question='Are these correct?')
    profile['slot_overrides']['counter']['aliases'] = {'counter_a': ['counter a'], 'counter_b': ['counter b']}
    profile['responses'] = {key: f'Diagnostic response: {key}.' for key in profile['responses']}
    profile['ui'] = dict(title='Counter preferences', note='A fictional diagnostic.',
                         state_title='Preferences', state_empty='No preferences yet.',
                         completion_text='Preferences confirmed. No external action occurred.')
    directory = tmp_path / 'configs/demo'
    directory.mkdir(parents=True)
    (directory / 'voice.json').write_bytes((ROOT / 'configs/demo/voice.json').read_bytes())
    profile_path = directory / 'service-desk.json'
    profile_path.write_text(json.dumps(profile), encoding='utf-8')
    config_path = tmp_path / 'configs/service-desk.json'
    config_path.write_text(json.dumps(base), encoding='utf-8')
    monkeypatch.setattr(demo_module, 'ROOT', tmp_path)
    monkeypatch.setattr(server_module, 'ROOT', tmp_path)
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'groq')
    def forbidden(*args, **kwargs):
        pytest.fail('Offline domain discovery/validation allocated a provider')
    monkeypatch.setattr(httpx, 'AsyncClient', forbidden)
    monkeypatch.setattr(httpx, 'Client', forbidden)
    for name in ('Router', 'SpeechToText', 'DemoVoice', 'SileroVAD'):
        monkeypatch.setattr(server_module, name, forbidden)
    return tmp_path, config_path, profile_path, base, profile


def response(**changes):
    result = dict(intent='task', ops=[], confidence=.99, unclear=False,
        is_affirmation=False, is_negation=False, clarification=None, proposed_ops=[],
        resolves_clarification=None, discard_clarification=None, discard_request=None)
    result.update(changes)
    return result


def test_new_flat_domain_loads_routes_corrects_and_confirms_without_registration(third_domain):
    _, path, _, _, _ = third_domain
    base = load_config(path)
    assert demo_module.demo_supported(base)
    config = demo_module.demo_config(base)
    state = ScopedTaskState(config)
    assert [slot['id'] for slot in config['slots']] == ['counter', 'date', 'time']
    assert demo_module.reply_text(state.next_action(), state.values, config['demo']) == 'Choose counter A or B.'
    body = response(ops=[{'op': 'set', 'slot': slot, 'value': value} for slot, value in
                         {'counter': 'counter_a', 'date': '2026-09-15', 'time': '10:30'}.items()])
    result = parse_response(json.dumps(body), config)
    assert state.consume(result.model_dump()).kind == 'readback'
    readback = demo_module.reply_text(Action('readback'), state.values, config['demo'])
    assert all(value in readback for value in ('Counter A', '15 / 09 / 2026', '10 : 30'))
    state.begin_confirmation(state.version)
    result = parse_response(json.dumps(response(ops=[{'op': 'set', 'slot': 'counter', 'value': 'counter_b'}],
                                                 is_affirmation=True)), config)
    assert state.consume(result.model_dump()).kind == 'readback' and not state.confirmed
    assert state.values == {'counter': 'counter_b', 'date': '2026-09-15', 'time': '10:30'}
    prompt = build_messages(config, state.router_context(), 'counter b')[0]['content'].lower()
    assert 'counter' in prompt and all(word not in prompt for word in ('pizza', 'doctor', 'clinic'))
    slots = response_format(config)['json_schema']['schema']['$defs']['Operation']['properties']['slot']['enum']
    assert slots == ['counter', 'date', 'time']
    state.begin_confirmation(state.version)
    assert state.consume(response(is_affirmation=True)).kind == 'accepted' and state.confirmed


@pytest.mark.parametrize('unsafe', ['../clinic', 'BadName', 'a/b'])
def test_unsafe_domain_ids_are_rejected_before_profile_lookup(third_domain, unsafe):
    _, _, _, base, _ = third_domain
    base = deepcopy(base)
    base['domain_id'] = unsafe
    with pytest.raises(ValueError, match='(?i)domain|safe|identifier|preview'):
        demo_module.demo_config(base)
    with pytest.raises(ValueError, match='(?i)domain|safe|identifier'):
        server_module.domain_config(unsafe)


def test_missing_custom_profile_is_not_advertised_as_supported(third_domain):
    _, path, profile_path, _, _ = third_domain
    # Rename the fixture profile within its own temporary directory.
    profile_path.rename(profile_path.with_suffix('.disabled'))
    config = load_config(path)
    assert not demo_module.demo_supported(config)
    with pytest.raises((ValueError, OSError), match='(?i)profile|preview|service-desk'):
        demo_module.demo_config(config)


def test_custom_collection_profile_fails_clearly_before_provider_allocation(third_domain):
    _, path, profile_path, _, profile = third_domain
    profile['state_kind'] = 'collection_scoped'
    profile_path.write_text(json.dumps(profile), encoding='utf-8')
    with pytest.raises(ValueError, match='(?i)flat|collection|adapter'):
        demo_module.demo_config(load_config(path))


def test_server_discovery_uses_valid_matching_config_files_and_stable_builtin_order(third_domain, monkeypatch):
    root, _, _, base, _ = third_domain
    for domain in ('pizza', 'clinic'):
        (root / 'configs' / f'{domain}.json').write_bytes((ROOT / 'configs' / f'{domain}.json').read_bytes())
    # A real schema file, invalid domain config, invalid JSON and alias filename
    # must not become selectable domains or prevent valid discovery.
    (root / 'configs/schema.json').write_bytes((ROOT / 'configs/schema.json').read_bytes())
    (root / 'configs/broken.json').write_text('{', encoding='utf-8')
    (root / 'configs/incomplete.json').write_text('{"domain_id":"incomplete"}', encoding='utf-8')
    (root / 'configs/alias.json').write_text(json.dumps(base), encoding='utf-8')
    with pytest.raises(ValueError, match='(?i)filename|match'):
        server_module.domain_config('alias')
    discovered = server_module.available_domains()
    assert [entry['id'] for entry in discovered] == ['pizza', 'clinic', 'service-desk']
    assert discovered[-1]['title'] == 'Counter service diagnostic'
    assert all(set(entry) == {'id', 'title'} for entry in discovered)
    checked = []
    async def fake_models(models, key):
        checked.append(models)
    monkeypatch.setenv('GROQ_API_KEY', 'offline-fixture')
    monkeypatch.setattr(server_module, 'check_models', fake_models)
    async def startup():
        async with server_module.lifespan(server_module.app):
            assert checked and len(checked[0]) >= 1
    asyncio.run(startup())


def test_builtin_pizza_still_uses_shared_voice_profile_filename(third_domain):
    root, _, _, _, _ = third_domain
    assert not (root / 'configs/demo/pizza.json').exists()
    config = demo_module.demo_config(load_config(ROOT / 'configs/pizza.json'))
    assert config['demo']['state_kind'] == 'collection_scoped'
    assert config['demo']['transaction_schema']['collection'] == 'items'


def test_flat_enum_list_named_items_uses_configured_flat_readback(third_domain):
    _, config_path, profile_path, base, profile = third_domain
    # Rename the field and every profile reference, while keeping enum values
    # counter_a/b. A generic list named items must not imply pizza rows.
    def rename_field(value):
        if isinstance(value, dict):
            return {('items' if key == 'counter' else key): rename_field(item)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [rename_field(item) for item in value]
        return 'items' if value == 'counter' else value
    base, profile = rename_field(base), rename_field(profile)
    next(slot for slot in base['slots'] if slot['id'] == 'items')['type'] = 'enum_list'
    config_path.write_text(json.dumps(base), encoding='utf-8')
    profile_path.write_text(json.dumps(profile), encoding='utf-8')
    config = demo_module.demo_config(load_config(config_path))
    state = ScopedTaskState(config)
    result = parse_response(json.dumps(response(ops=[
        {'op': 'set', 'slot': 'items', 'value': ['counter_a', 'counter_b']},
        {'op': 'set', 'slot': 'date', 'value': '2026-09-15'},
        {'op': 'set', 'slot': 'time', 'value': '10:30'},
    ])), config)
    assert state.consume(result.model_dump()).kind == 'readback'
    assert state.values['items'] == ['counter_a', 'counter_b'] and state.ready
    text = demo_module.reply_text(Action('readback'), state.values, config['demo'])
    assert all(value in text for value in ('Counter A', 'Counter B', '15 / 09 / 2026', '10 : 30'))
