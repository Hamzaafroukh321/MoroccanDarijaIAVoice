"""Validate the actual provider JSON Schema semantically, without inference."""
from copy import deepcopy
import json

from jsonschema import Draft202012Validator
import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.router import parse_response, response_format


def payload(clarification=None):
    kind = clarification['kind'] if clarification is not None else None
    if kind in {'ambiguous_value', 'unsupported_value', 'unintelligible'}:
        clarification = {'coupled_slots': None, **clarification}
    return dict(intent={'ambiguous_value': 'ambiguous', 'unsupported_value': 'out_of_scope',
        'unintelligible': 'unclear', 'order_details': 'ambiguous'}.get(kind, 'task'),
        ops=[], proposed_ops=[], confidence=.99, unclear=kind == 'unintelligible',
        is_affirmation=False, is_negation=False, clarification=clarification,
        resolves_clarification=None, discard_clarification=None, discard_request=None)


@pytest.fixture(params=['clinic', 'custom'])
def case(request, monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    config = demo_config(load_config(ROOT / 'configs/clinic.json'))
    field = 'doctor'
    if request.param == 'custom':
        config = json.loads(json.dumps(config).replace('doctor', 'counter'))
        config['domain_id'] = 'service-desk'
        field = 'counter'
    schema = response_format(config)['json_schema']['schema']
    Draft202012Validator.check_schema(schema)
    return config, field, Draft202012Validator(schema)


@pytest.mark.parametrize('kind', ['ambiguous_value', 'unsupported_value'])
def test_value_clarification_requires_nonnull_configured_scope(case, kind):
    config, field, validator = case
    for slot in (field, 'date', 'time'):
        body = payload({'kind': kind, 'slot': slot, 'item_ids': []})
        validator.validate(body)
        assert parse_response(json.dumps(body), config).clarification.slot == slot
    for slot in (None, 'unconfigured', 7, True, ''):
        body = payload({'kind': kind, 'slot': slot, 'item_ids': []})
        assert not validator.is_valid(body)
        with pytest.raises(ValueError):
            parse_response(json.dumps(body), config)


def test_unintelligible_allows_null_or_configured_scope_and_outer_null_stays_valid(case):
    config, field, validator = case
    for slot in (None, field, 'date', 'time'):
        body = payload({'kind': 'unintelligible', 'slot': slot, 'item_ids': []})
        validator.validate(body)
        assert parse_response(json.dumps(body), config).clarification.slot == slot
    validator.validate(payload())
    assert parse_response(json.dumps(payload()), config).clarification is None
    assert not validator.is_valid(payload({'kind': 'unintelligible', 'slot': 'unconfigured', 'item_ids': []}))


@pytest.mark.parametrize('kind', ['ambiguous_value', 'unsupported_value', 'unintelligible'])
def test_every_scope_branch_is_closed_and_requires_all_fields(case, kind):
    _, field, validator = case
    good = payload({'kind': kind, 'slot': field, 'item_ids': []})
    validator.validate(good)
    for key in ('kind', 'slot', 'item_ids', 'coupled_slots'):
        bad = deepcopy(good)
        del bad['clarification'][key]
        assert not validator.is_valid(bad), key
    bad = deepcopy(good)
    bad['clarification']['unrecognized'] = 'must not leak through a branch'
    assert not validator.is_valid(bad)
    for ids in ([1], [True], ['1'], None):
        bad = deepcopy(good)
        bad['clarification']['item_ids'] = ids
        assert not validator.is_valid(bad)
    bad = deepcopy(good)
    bad['clarification']['kind'] = 'invented_kind'
    assert not validator.is_valid(bad)
    for key in good:
        bad = deepcopy(good)
        del bad[key]
        assert not validator.is_valid(bad), key
    bad = deepcopy(good)
    bad['invented_top_level_field'] = True
    assert not validator.is_valid(bad)


def test_pizza_and_base_schema_keep_their_distinct_contracts(monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    base = load_config(ROOT / 'configs/pizza.json')
    config = demo_config(base)
    schema = response_format(config)['json_schema']['schema']
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    legal = payload({'kind': 'order_details', 'slot': None, 'item_ids': []})
    validator.validate(legal)
    assert parse_response(json.dumps(legal), config).clarification.slot is None
    assert not validator.is_valid(payload({'kind': 'ambiguous_value', 'slot': None, 'item_ids': []}))
    base_schema = response_format(base)['json_schema']['schema']
    Draft202012Validator.check_schema(base_schema)
    base_validator = Draft202012Validator(base_schema)
    research = dict(ops=[], confidence=.99, unclear=True, is_affirmation=False, is_negation=False)
    base_validator.validate(research)
    assert not base_validator.is_valid({**research, 'clarification': None})


def test_coupled_schema_tracks_configured_fields_only_for_ambiguity(case):
    config, field, validator = case
    legal = payload({'kind': 'ambiguous_value', 'slot': field, 'item_ids': [], 'coupled_slots': ['time']})
    validator.validate(legal)
    assert parse_response(json.dumps(legal), config).clarification.coupled_slots == ['time']
    for names in (['unknown'], [1], ['time', 'date', field]):
        bad = deepcopy(legal)
        bad['clarification']['coupled_slots'] = names
        assert not validator.is_valid(bad)
    for kind in ('unsupported_value', 'unintelligible'):
        bad = payload({'kind': kind, 'slot': field, 'item_ids': [], 'coupled_slots': ['time']})
        assert not validator.is_valid(bad)
    legacy = deepcopy(legal)
    del legacy['clarification']['coupled_slots']
    assert not validator.is_valid(legacy)
    assert parse_response(json.dumps(legacy), config).clarification.coupled_slots is None
