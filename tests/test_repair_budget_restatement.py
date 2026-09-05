"""Accepted restatements recover a repair budget without bypassing questions."""

from copy import deepcopy
import json

import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.demo_order import DemoOrderState
from engine.router import parse_response
from engine.scoped_task import ScopedTaskState


def consume_state(state, **changes):
    response = dict(intent='task', ops=[], confidence=.99, unclear=False,
                    is_affirmation=False, is_negation=False, clarification=None,
                    resolves_clarification=None, proposed_ops=[],
                    discard_clarification=None, discard_request=None)
    response.update(changes)
    return state.consume(parse_response(json.dumps(response), state.config).model_dump())


@pytest.fixture(params=['pizza', 'clinic'])
def dialogue(request):
    domain = request.param
    config = demo_config(load_config(ROOT / 'configs' / f'{domain}.json'))
    config['demo']['max_repairs'] = 2
    state = DemoOrderState(config) if domain == 'pizza' else ScopedTaskState(config)

    def operation(slot, value):
        result = dict(op='set', slot=slot, value=value)
        if domain == 'pizza':
            result['item_id'] = None if slot == 'drink' else 1
        return result

    if domain == 'pizza':
        complete = [dict(op='create', item_id=1, slot=None, value=None),
                    operation('quantity', 1), operation('size', 'large'),
                    operation('toppings', ['cheese']), operation('drink', ['water'])]
        restatement = operation('size', 'large')
        scope_answer = operation('drink', ['water'])
        question = dict(kind='unsupported_drink', slot='drink', item_ids=[])
        ambiguous = dict(kind='order_details', slot='size', item_ids=[1])
        invalid = operation('quantity', 999)
    else:
        complete = [operation('doctor', 'doctor_a'), operation('date', '2026-09-15'),
                    operation('time', '10:30')]
        restatement = operation('time', '10:30')
        scope_answer = operation('doctor', 'doctor_a')
        question = dict(kind='unsupported_value', slot='doctor', item_ids=[])
        ambiguous = dict(kind='ambiguous_value', slot='time', item_ids=[])
        invalid = operation('time', '99:99')

    def consume(**changes):
        return consume_state(state, **changes)

    assert consume(ops=complete).kind == 'readback'
    return dict(state=state, consume=consume, restatement=restatement,
                scope_answer=scope_answer, question=question, ambiguous=ambiguous,
                invalid=invalid, domain=domain, operation=operation)


@pytest.mark.parametrize('with_agreement', [False, True])
def test_valid_restatement_recovers_budget_and_still_requires_new_readback(dialogue, with_agreement):
    state, consume = dialogue['state'], dialogue['consume']
    committed, version = deepcopy(state.values), state.version
    state.begin_confirmation(version)
    assert consume(is_negation=True).kind == 'repair'
    assert consume(is_negation=True).kind == 'repair'
    assert state.repair_count == 2

    assert consume(ops=[dialogue['restatement']], is_affirmation=with_agreement).kind == 'readback'
    assert state.repair_count == 0 and not state.awaiting_correction
    assert state.values == committed and state.version == version
    assert state.readback_version is None and not state.confirmed
    assert consume(is_affirmation=True).kind == 'readback'
    assert not state.confirmed

    state.begin_confirmation(state.version)
    # Previously this became handoff with count 3 despite the accepted repair.
    assert consume(is_negation=True).kind == 'repair'
    assert state.repair_count == 1 and not state.confirmed


@pytest.mark.parametrize('kind', ['unsupported', 'ambiguous'])
def test_unrelated_restatement_does_not_reset_an_unresolved_question(dialogue, kind):
    state, consume = dialogue['state'], dialogue['consume']
    committed = deepcopy(state.values)
    question = dialogue['question'] if kind == 'unsupported' else dialogue['ambiguous']
    unrelated = dialogue['restatement'] if kind == 'unsupported' else dialogue['scope_answer']
    intent = 'out_of_scope' if kind == 'unsupported' else 'ambiguous'
    assert consume(intent=intent, clarification=question).kind == question['kind']
    pending = deepcopy(state.pending_clarification)
    assert state.repair_count == 1

    assert consume(ops=[unrelated]).kind == question['kind']
    assert state.repair_count == 2 and state.awaiting_correction
    assert state.pending_clarification == pending and state.values == committed
    assert state.readback_version is None and not state.confirmed
    assert consume(ops=[unrelated]).kind == 'handoff'


def test_stale_resolution_cannot_reset_budget_or_modify_state(dialogue):
    state, consume = dialogue['state'], dialogue['consume']
    consume(intent='out_of_scope', clarification=dialogue['question'])
    pending, committed = deepcopy(state.pending_clarification), deepcopy(state.values)
    with pytest.raises(ValueError, match='stale'):
        consume(ops=[dialogue['scope_answer']], resolves_clarification=pending['id'] + 1)
    assert state.repair_count == 1 and state.pending_clarification == pending
    assert state.values == committed and not state.confirmed


def test_invalid_field_restatement_cannot_reset_budget(dialogue):
    state, consume = dialogue['state'], dialogue['consume']
    consume(is_negation=True)
    committed = deepcopy(state.values)
    with pytest.raises(ValueError):
        consume(ops=[dialogue['invalid']])
    assert state.repair_count == 1 and state.awaiting_correction
    assert state.values == committed and not state.confirmed


def test_negated_unchanged_field_is_still_an_unresolved_repair(dialogue):
    state, consume = dialogue['state'], dialogue['consume']
    consume(is_negation=True)
    assert consume(ops=[dialogue['restatement']], is_negation=True).kind == 'repair'
    assert state.repair_count == 2 and state.awaiting_correction
    assert not state.confirmed


@pytest.mark.parametrize('kind', ['clear', 'remove'])
def test_removing_an_absent_field_cannot_reset_budget_or_clear_correction(dialogue, kind):
    state = type(dialogue['state'])(dialogue['state'].config)
    slot = 'drink' if dialogue['domain'] == 'pizza' else 'date'
    operation = dialogue['operation'](slot, None)
    operation['op'] = kind
    before, version = deepcopy(state.values), state.version
    assert consume_state(state, is_negation=True).kind == 'repair'
    assert consume_state(state, is_negation=True).kind == 'repair'
    assert consume_state(state, ops=[operation]).kind == 'handoff'
    assert state.repair_count == 3 and state.awaiting_correction
    assert state.values == before and state.version == version


def test_removing_an_unselected_value_cannot_recover_rejected_summary(dialogue):
    state, consume = dialogue['state'], dialogue['consume']
    operation = (dialogue['operation']('toppings', ['mushroom']) if dialogue['domain'] == 'pizza'
                 else dialogue['operation']('date', '2026-09-16'))
    operation['op'] = 'remove'
    before, version = deepcopy(state.values), state.version
    consume(is_negation=True)
    consume(is_negation=True)
    assert consume(ops=[operation]).kind == 'handoff'
    assert state.repair_count == 3 and state.awaiting_correction
    assert state.values == before and state.version == version


def test_repeating_known_field_does_not_answer_different_missing_field(dialogue):
    state = type(dialogue['state'])(dialogue['state'].config)
    if dialogue['domain'] == 'pizza':
        known = dialogue['operation']('size', 'large')
        initial = [dict(op='create', item_id=1, slot=None, value=None), known]
    else:
        known = dialogue['operation']('doctor', 'doctor_a')
        initial = [known]
    consume_state(state, ops=initial)
    consume_state(state, unclear=True)
    assert not state.ready and state.repair_count == 1
    action = consume_state(state, ops=[known])
    assert action.kind == 'ask' and action.slot != known['slot']
    assert state.repair_count == 1


def test_empty_add_does_not_recover_a_rejected_pizza_summary():
    config = demo_config(load_config(ROOT / 'configs' / 'pizza.json'))
    config['demo']['max_repairs'] = 2
    state = DemoOrderState(config)
    fields = dict(quantity=1, size='large', toppings=['cheese'], drink=['water'])
    initial = [dict(op='create', item_id=1, slot=None, value=None)] + [
        dict(op='set', item_id=None if slot == 'drink' else 1, slot=slot, value=value)
        for slot, value in fields.items()]
    consume_state(state, ops=initial)
    operation = dict(op='add', item_id=1, slot='toppings', value=[])
    consume_state(state, is_negation=True)
    consume_state(state, is_negation=True)
    assert consume_state(state, ops=[operation]).kind == 'handoff'
    assert state.repair_count == 3 and state.awaiting_correction


def test_create_then_delete_cannot_recover_budget_by_advancing_only_item_identity():
    config = demo_config(load_config(ROOT / 'configs' / 'pizza.json'))
    config['demo']['max_repairs'] = 2
    state = DemoOrderState(config)
    before, version = deepcopy(state.values), state.version
    consume_state(state, is_negation=True)
    consume_state(state, is_negation=True)
    operations = [dict(op=kind, item_id=1, slot=None, value=None)
                  for kind in ['create', 'delete']]

    assert consume_state(state, ops=operations).kind == 'handoff'
    assert state.values == before
    # The allocator's identity guarantee is independent of dialogue progress.
    assert state.version > version and state.next_item_id == 2
    assert state.repair_count == 3 and state.awaiting_correction
    assert state.readback_version is None and not state.confirmed
