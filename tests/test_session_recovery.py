"""Process-local resume tokens preserve committed facts without old dialogue."""
from copy import deepcopy
import json

import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.demo_order import DemoOrderState
from engine.scoped_task import ScopedTaskState


def operation(slot, value, item_id=None):
    result = {'op': 'set', 'slot': slot, 'value': value}
    if item_id is not None:
        result['item_id'] = item_id
    return result


def response(**changes):
    result = dict(intent='task', ops=[], unclear=False, is_affirmation=False,
        is_negation=False, clarification=None, proposed_ops=[], resolves_clarification=None,
        discard_clarification=None, discard_request=None)
    result.update(changes)
    return result


@pytest.fixture(params=['clinic', 'custom', 'pizza'])
def case(request, monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    domain = request.param
    config = demo_config(load_config(ROOT / f"configs/{'pizza' if domain == 'pizza' else 'clinic'}.json"))
    field = 'doctor'
    if domain == 'custom':
        config = json.loads(json.dumps(config).replace('doctor', 'counter'))
        config['domain_id'] = 'service-desk'
        field = 'counter'
    state = DemoOrderState(config) if domain == 'pizza' else ScopedTaskState(config)
    if domain == 'pizza':
        ops = []
        for item_id in range(1, 8):
            ops.append({'op': 'create', 'item_id': item_id, 'slot': None, 'value': None})
            if item_id not in (2, 7):
                ops.append({'op': 'delete', 'item_id': item_id, 'slot': None, 'value': None})
            else:
                ops += [operation('quantity', 1, item_id), operation('size', 'small' if item_id == 2 else 'large', item_id),
                        operation('toppings', [] if item_id == 2 else ['cheese'], item_id)]
        ops.append({'op': 'set', 'item_id': None, 'slot': 'drink', 'value': ['water']})
        state.apply(ops)
        expected = deepcopy(state.values)
        for index, item in enumerate(expected['items'], 1):
            item['id'] = index
        partial_ops = [{'op': 'clear', 'item_id': 7, 'slot': 'size', 'value': None}]
        pending = response(intent='out_of_scope', clarification={
            'kind': 'unsupported_drink', 'slot': 'drink', 'item_ids': []}, proposed_ops=[
            operation('quantity', 4, 2)])
    else:
        state.apply([operation(field, field + '_a'), operation('date', '2026-09-15'), operation('time', '11:30')])
        expected = deepcopy(state.values)
        partial_ops = [{'op': 'clear', 'slot': 'time', 'value': None}]
        pending = response(intent='ambiguous', clarification={
            'kind': 'ambiguous_value', 'slot': field, 'item_ids': []}, proposed_ops=[operation('time', '15:30')])
    return dict(config=config, state=state, expected=expected, partial_ops=partial_ops, pending=pending, domain=domain)


def store(clock=None, **kwargs):
    from engine.recovery import RecoveryStore
    return RecoveryStore(clock=clock or (lambda: 100.0), **kwargs)


@pytest.mark.parametrize('partial', [False, True])
def test_restoration_uses_committed_values_only_and_starts_unconfirmed(case, partial):
    state, config = case['state'], case['config']
    if partial:
        state.apply(case['partial_ops'])
    expected = deepcopy(state.values)
    if case['domain'] == 'pizza':
        for index, item in enumerate(expected['items'], 1):
            item['id'] = index
    state.begin_confirmation(state.version)
    state.consume(case['pending'])
    old = deepcopy(state.values)
    registry = store()
    offer = registry.offer(config, state, 'source-session', pending_request={
        'id': 'request_1', 'text': 'private abandoned original'})
    assert offer['domain_id'] == config['domain_id']
    assert offer['expires_in_seconds'] == 1800 and offer['discarded_pending'] is True
    recovered = registry.restore(offer['token'], config)
    fresh = recovered.task
    assert fresh is not state and fresh.values == expected
    assert state.values == old
    assert fresh.pending_proposal is None and fresh.pending_clarification is None
    assert fresh.readback_version is None and fresh.confirmed_version is None and not fresh.confirmed
    assert recovered.discarded_pending is True and recovered.source_session_id == 'source-session'
    assert fresh.next_action().kind == ('ask' if partial else 'readback')
    if case['domain'] == 'pizza':
        assert fresh.next_item_id == 3
        fresh.apply([{'op': 'create', 'item_id': 3, 'slot': None, 'value': None}])
        assert [item['id'] for item in fresh.values['items']] == [1, 2, 3]
    with pytest.raises(ValueError):
        registry.restore(offer['token'], config)


def test_offers_and_restores_do_not_alias_caller_owned_values(case):
    registry = store()
    config, state = case['config'], case['state']
    offered = registry.offer(config, state, 'source-session')
    assert offered['slots'] == case['expected']
    offered['slots'].clear()
    state.values.clear()
    fresh = registry.restore(offered['token'], config).task
    assert fresh.values == case['expected']


def test_wrong_domain_or_configuration_preserves_valid_token(case):
    registry = store()
    config = case['config']
    offered = registry.offer(config, case['state'], 'source-session')
    wrong_domain = deepcopy(config)
    wrong_domain['domain_id'] = 'other-domain'
    changed = deepcopy(config)
    first_field = next(iter(changed['demo']['questions']))
    changed['demo']['questions'][first_field] += ' changed'
    for incompatible in (wrong_domain, changed):
        with pytest.raises(ValueError):
            registry.restore(offered['token'], incompatible)
    assert registry.restore(offered['token'], config).task.values == case['expected']


@pytest.mark.parametrize('token', [None, '', 7, True, [], '../token', 'a' * 10000])
def test_malformed_tokens_fail_without_mutating_valid_recovery(case, token):
    registry = store()
    offered = registry.offer(case['config'], case['state'], 'source-session')
    with pytest.raises(ValueError):
        registry.restore(token, case['config'])
    assert registry.restore(offered['token'], case['config']).task.values == case['expected']


def test_expiry_and_capacity_are_bounded_in_process(case):
    clock = [100.0]
    registry = store(clock=lambda: clock[0])
    offered = registry.offer(case['config'], case['state'], 'source-session')
    clock[0] += 1800
    with pytest.raises(ValueError):
        registry.restore(offered['token'], case['config'])
    offers = [registry.offer(case['config'], case['state'], f'source-{index}') for index in range(33)]
    assert len({offer['token'] for offer in offers}) == 33
    with pytest.raises(ValueError):
        registry.restore(offers[0]['token'], case['config'])
    for offer in offers[1:]:
        assert registry.restore(offer['token'], case['config']).task.values == case['expected']


def test_empty_or_invalid_committed_state_does_not_create_an_offer(case):
    registry = store()
    state = case['state']
    state.values.clear()
    assert registry.offer(case['config'], state, 'empty-session') is None
    state.values['unconfigured'] = 'must not resume'
    assert registry.offer(case['config'], state, 'invalid-session') is None
