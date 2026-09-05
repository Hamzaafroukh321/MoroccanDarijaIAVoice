"""Canonical state-machine traces, not claims about ASR or Darija accuracy."""

from copy import deepcopy
import json

import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.demo_order import DemoOrderState
from engine.router import parse_response
from engine.scoped_task import ScopedTaskState


def payload(**updates):
    response = dict(ops=[], confidence=.95, unclear=False, is_affirmation=False,
                    is_negation=False, intent='task', clarification=None,
                    resolves_clarification=None, proposed_ops=[], discard_clarification=None)
    response.update(updates)
    return response


@pytest.fixture(params=['pizza', 'clinic'])
def dialogue(request):
    config = demo_config(load_config(ROOT / 'configs' / f'{request.param}.json'))
    if request.param == 'pizza':
        state = DemoOrderState(config)

        def op(kind, slot=None, value=None, item_id=None):
            return dict(op=kind, slot=slot, value=value, item_id=item_id)

        complete = [op('create', item_id=1), op('set', 'quantity', 1, 1),
                    op('set', 'size', 'large', 1), op('set', 'toppings', [], 1),
                    op('set', 'drink', ['cola'])]
        proposed = [op('set', 'size', 'small', 1)]
        clarification = dict(kind='unsupported_drink', slot='drink', item_ids=[])
        resolution = [op('set', 'drink', ['water'])]
        redundant = [op('set', 'size', 'large', 1)]
    else:
        state = ScopedTaskState(config)

        def op(kind, slot, value):
            return dict(op=kind, slot=slot, value=value)

        complete = [op('set', 'doctor', 'doctor_a'), op('set', 'date', '2026-09-15'),
                    op('set', 'time', '10:30')]
        proposed = [op('set', 'time', '11:30')]
        clarification = dict(kind='unsupported_value', slot='doctor', item_ids=[])
        resolution = [op('set', 'doctor', 'doctor_b')]
        redundant = [op('set', 'time', '10:30')]

    def consume(**updates):
        parsed = parse_response(json.dumps(payload(**updates)), config).model_dump()
        return state.consume(parsed)

    assert consume(ops=complete).kind == 'readback'
    assert state.ready
    return dict(state=state, config=config, consume=consume, proposed=proposed,
                clarification=clarification, resolution=resolution, redundant=redundant)


def test_bare_no_then_repeated_yes_has_bounded_repair(dialogue):
    state, consume = dialogue['state'], dialogue['consume']
    before = deepcopy(state.values)
    state.begin_confirmation(state.version)
    assert consume(is_negation=True).kind == 'repair'
    assert state.awaiting_correction and state.readback_version is None
    actions = [consume(is_affirmation=True).kind
               for _ in range(dialogue['config']['demo']['max_repairs'])]
    assert 'accepted' not in actions and not state.confirmed
    assert state.values == before
    assert actions[-1] == 'handoff', 'Repeated yes must not loop forever while an explicit correction is still awaited.'


def test_no_then_redundant_correction_requires_new_readback_and_yes(dialogue):
    state, consume = dialogue['state'], dialogue['consume']
    state.begin_confirmation(state.version)
    assert consume(is_negation=True).kind == 'repair'
    action = consume(ops=dialogue['redundant'], is_affirmation=True)
    assert action.kind == 'readback'
    assert not state.awaiting_correction
    assert state.readback_version is None and not state.confirmed
    assert consume(is_affirmation=True).kind == 'readback'
    assert not state.confirmed
    state.begin_confirmation(state.version)
    assert consume(is_affirmation=True).kind == 'accepted'


def test_valid_changed_correction_resets_repair_budget(dialogue):
    state, consume = dialogue['state'], dialogue['consume']
    before = deepcopy(state.values)
    state.begin_confirmation(state.version)
    assert consume(is_negation=True).kind == 'repair'
    assert consume(is_affirmation=True).kind == 'repair'
    assert state.repair_count == 2
    assert consume(ops=dialogue['proposed']).kind == 'readback'
    assert state.values != before
    assert state.repair_count == 0 and not state.awaiting_correction
    assert state.readback_version is None and not state.confirmed
    # The new valid turn earns a fresh budget; it does not consume an old ACK.
    state.begin_confirmation(state.version)
    assert consume(is_negation=True).kind == 'repair'
    assert state.repair_count == 1


def test_redundant_restatement_does_not_escape_scoped_question(dialogue):
    state, consume = dialogue['state'], dialogue['consume']
    before = deepcopy(state.values)
    state.begin_confirmation(state.version)
    consume(intent='out_of_scope', clarification=dialogue['clarification'])
    pending = deepcopy(state.pending_clarification)
    action = consume(ops=dialogue['redundant'], is_affirmation=True)
    assert action.kind == pending['kind']
    assert state.pending_clarification == pending
    assert state.awaiting_correction and state.readback_version is None
    assert state.values == before and not state.confirmed


def test_discard_then_new_proposal_rejects_old_token_and_requires_fresh_readback(dialogue):
    state, consume = dialogue['state'], dialogue['consume']
    before, original_version = deepcopy(state.values), state.version
    state.begin_confirmation(original_version)
    consume(intent='out_of_scope', clarification=dialogue['clarification'],
            proposed_ops=dialogue['proposed'])
    first_id = state.pending_clarification['id']
    assert state.values == before
    assert consume(discard_clarification=first_id).kind == 'readback'
    assert state.values == before and state.version > original_version
    assert state.pending_proposal is None and state.pending_clarification is None
    assert state.readback_version is None and not state.confirmed
    assert consume(is_affirmation=True).kind == 'readback'
    consume(intent='out_of_scope', clarification=dialogue['clarification'],
            proposed_ops=dialogue['proposed'])
    second_id = state.pending_clarification['id']
    assert second_id > first_id
    pending = deepcopy(state.pending_proposal)
    for updates in ({'discard_clarification': first_id},
                    {'resolves_clarification': first_id, 'ops': dialogue['resolution']}):
        with pytest.raises(ValueError, match='stale'):
            consume(**updates)
        assert state.pending_proposal == pending and state.values == before
    assert consume(resolves_clarification=second_id, ops=dialogue['resolution'],
                   is_affirmation=True).kind == 'readback'
    assert state.values != before and state.pending_proposal is None
    assert not state.confirmed and state.readback_version is None
    assert consume(is_affirmation=True).kind == 'readback'
    state.begin_confirmation(state.version)
    assert consume(is_affirmation=True).kind == 'accepted'


def test_unrelated_task_turn_cannot_commit_pending_proposal(dialogue):
    state, consume = dialogue['state'], dialogue['consume']
    before = deepcopy(state.values)
    consume(intent='out_of_scope', clarification=dialogue['clarification'],
            proposed_ops=dialogue['proposed'])
    pending = deepcopy(state.pending_proposal)
    with pytest.raises(ValueError, match='staged proposal'):
        consume(ops=dialogue['redundant'])
    assert state.values == before and state.pending_proposal == pending
    assert not state.confirmed
