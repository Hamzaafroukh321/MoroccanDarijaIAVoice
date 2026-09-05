"""Domain-neutral collection state behavior with temporary equipment preferences."""
from copy import deepcopy

import pytest

from collection_fixtures import collection_config, write_collection_fixture
from engine.config import load_config
from engine.collection_schema import validate_collection_schema
from engine.collection_task import ConfiguredCollectionState


@pytest.fixture(params=[False, True], ids=['equipment', 'renamed'])
def case(request, monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    config, names = collection_config(renamed=request.param)
    return config, names, ConfiguredCollectionState(config)


def operation(op, item_id=None, slot=None, value=None):
    return dict(op=op, item_id=item_id, slot=slot, value=value)


def row(names, item_id, asset='camera', quantity=1):
    return [operation('create', item_id), operation('set', item_id, names['asset'], asset),
            operation('set', item_id, names['quantity'], quantity)]


def response(**changes):
    payload = dict(intent='task', ops=[], confidence=.99, unclear=False,
        is_affirmation=False, is_negation=False, clarification=None, proposed_ops=[],
        resolves_clarification=None, discard_clarification=None, discard_request=None)
    payload.update(changes)
    return payload


def proposed_question(names):
    return response(intent='out_of_scope', clarification=dict(kind='unsupported_value',
        slot=names['asset'], item_ids=[1]), proposed_ops=[operation('create', 1),
        operation('set', 1, names['quantity'], 2), operation('set', None, names['date'], '2026-09-15')])


def test_complete_fixture_base_is_schema_valid_and_remains_unreviewed(case, tmp_path):
    _, names, _ = case
    files = write_collection_fixture(tmp_path, renamed=names['domain'] == 'kit-hire')
    loaded = load_config(files['config_path'])
    assert loaded['domain_id'] == names['domain']
    assert loaded['needs_human_review'] is True
    assert files['profile']['evaluation_eligible'] is False
    assert files['profile']['state_kind'] == 'configured_collection_scoped'


def test_explicit_rows_correct_only_target_and_ids_are_never_reused(case):
    _, names, task = case
    action = task.next_action()
    assert action.kind == 'ask' and action.slot == names['asset'] and action.item_id == 1
    assert task.values == {names['collection']: []} and task.next_item_id == 1
    task.apply(row(names, 1) + row(names, 2, 'tripod', 2) +
               [operation('set', None, names['date'], '2026-09-15')])
    assert task.ready and task.next_item_id == 3
    first = deepcopy(task.values[names['collection']][0])
    task.apply([operation('set', 2, names['quantity'], 4)])
    assert task.values[names['collection']][0] == first
    assert task.values[names['collection']][1][names['quantity']] == 4
    task.apply([operation('delete', 1)] + row(names, 3, 'camera', 3))
    assert [item['id'] for item in task.values[names['collection']]] == [2, 3]
    assert task.next_item_id == 4


@pytest.mark.parametrize('invalid', ['unknown_row', 'boolean_id', 'bad_value', 'root_row_id', 'item_no_id', 'extra_key'])
def test_invalid_late_operation_rolls_back_rows_values_and_allocator(case, invalid):
    _, names, task = case
    task.apply(row(names, 1))
    bad = operation('set', 1, names['quantity'], 2)
    if invalid == 'unknown_row':
        bad['item_id'] = 99
    elif invalid == 'boolean_id':
        bad['item_id'] = True
    elif invalid == 'bad_value':
        bad['value'] = 11
    elif invalid == 'root_row_id':
        bad = operation('set', 1, names['date'], '2026-09-15')
    elif invalid == 'item_no_id':
        bad['item_id'] = None
    else:
        bad['collection'] = names['collection']
    before = deepcopy(task.__dict__)
    with pytest.raises(ValueError):
        task.apply(row(names, 2, 'tripod', 2) + [bad])
    assert task.__dict__ == before


def test_optional_item_field_does_not_block_but_root_date_still_required(case):
    config, names, _ = case
    next(slot for slot in config['slots'] if slot['id'] == names['quantity'])['required'] = False
    config['demo']['required_slots'].remove(names['quantity'])
    task = ConfiguredCollectionState(config)
    task.apply([operation('create', 1), operation('set', 1, names['asset'], 'camera')])
    assert not task.ready and task.next_action().slot == names['date']
    assert task.next_action().item_id is None
    task.apply([operation('set', None, names['date'], '2026-09-15')])
    assert task.ready and task.next_action().kind == 'readback'


def test_required_empty_list_is_unknown_not_a_pizza_plain_special_case(case):
    config, names, _ = case
    next(slot for slot in config['slots'] if slot['id'] == names['asset'])['type'] = 'enum_list'
    task = ConfiguredCollectionState(config)
    task.apply([operation('create', 1), operation('set', 1, names['asset'], []),
        operation('set', 1, names['quantity'], 1), operation('set', None, names['date'], '2026-09-15')])
    assert task.values[names['collection']][0][names['asset']] == []
    assert not task.ready
    assert task.next_action().slot == names['asset'] and task.next_action().item_id == 1
    task.apply([operation('add', 1, names['asset'], 'camera')])
    assert task.ready


def test_proposed_new_row_question_validates_preview_id_then_commits_once(case):
    _, names, task = case
    task.consume(proposed_question(names))
    assert task.values == {names['collection']: []} and task.next_item_id == 1
    preview = task.pending_proposal
    assert preview['next_item_id'] == 2
    assert preview['state'] == {names['collection']: [{'id': 1, names['quantity']: 2}], names['date']: '2026-09-15'}
    assert task.pending_clarification['item_ids'] == [1]
    assert task.next_action().item_id == 1
    question_id = task.pending_clarification['id']
    action = task.consume(response(ops=[operation('set', 1, names['asset'], 'tripod')], resolves_clarification=question_id))
    assert action.kind == 'readback' and task.ready
    assert task.next_item_id == 2 and len(task.values[names['collection']]) == 1
    assert task.values[names['collection']][0] == {'id': 1, names['quantity']: 2, names['asset']: 'tripod'}
    assert task.pending_proposal is None and task.pending_clarification is None
    assert task.readback_version is None and not task.confirmed
    with pytest.raises(ValueError):
        task.consume(response(ops=[operation('set', 1, names['asset'], 'tripod')], resolves_clarification=question_id))
    assert task.consume(response(is_affirmation=True)).kind == 'readback'
    task.begin_confirmation(task.version)
    assert task.consume(response(is_affirmation=True)).kind == 'accepted'


@pytest.mark.parametrize('invalid', ['unknown_preview_id', 'unresolved_value'])
def test_invalid_proposal_is_atomic_before_opening_question(case, invalid):
    _, names, task = case
    candidate = proposed_question(names)
    if invalid == 'unknown_preview_id':
        candidate['clarification']['item_ids'] = [2]
    else:
        candidate['proposed_ops'].append(operation('set', 1, names['asset'], 'camera'))
    before = deepcopy(task.__dict__)
    with pytest.raises(ValueError):
        task.consume(candidate)
    assert task.__dict__ == before


def test_wrong_scope_and_missing_resolution_id_preserve_draft(case):
    _, names, task = case
    task.consume(proposed_question(names))
    before = deepcopy(task.__dict__)
    question_id = task.pending_clarification['id']
    for candidate in [response(ops=[operation('set', 1, names['asset'], 'camera')]),
        response(ops=[operation('set', None, names['date'], '2026-09-16')], resolves_clarification=question_id),
        response(ops=[operation('set', 1, names['asset'], 'camera')], resolves_clarification=question_id + 1)]:
        with pytest.raises(ValueError):
            task.consume(candidate)
        assert task.__dict__ == before


def test_root_question_can_hold_multiple_rows_without_losing_independent_facts(case):
    _, names, task = case
    task.consume(response(intent='ambiguous', clarification=dict(kind='ambiguous_value',
        slot=names['date'], item_ids=[]), proposed_ops=row(names, 1) + row(names, 2, 'tripod', 3)))
    assert task.values == {names['collection']: []} and task.next_item_id == 1
    question_id = task.pending_clarification['id']
    task.consume(response(ops=[operation('set', None, names['date'], '2026-09-15')], resolves_clarification=question_id))
    assert task.ready and task.next_item_id == 3
    assert task.values[names['collection']] == [
        {'id': 1, names['asset']: 'camera', names['quantity']: 1},
        {'id': 2, names['asset']: 'tripod', names['quantity']: 3}]


def test_discard_preserves_committed_rows_and_does_not_spend_draft_ids(case):
    _, names, task = case
    task.apply(row(names, 1) + [operation('set', None, names['date'], '2026-09-15')])
    original, next_id, version = deepcopy(task.values), task.next_item_id, task.version
    candidate = proposed_question(names)
    candidate['clarification']['item_ids'] = [2]
    for change in candidate['proposed_ops']:
        if change['item_id'] is not None:
            change['item_id'] = 2
    task.consume(candidate)
    assert task.pending_proposal['next_item_id'] == 3 and task.next_item_id == next_id
    task.consume(response(discard_clarification=task.pending_clarification['id']))
    assert task.values == original and task.next_item_id == next_id
    assert task.version == version + 1 and task.readback_version is None
    task.apply(row(names, 2, 'tripod', 1))
    assert [item['id'] for item in task.values[names['collection']]] == [1, 2]


def test_bare_confirmation_cannot_resolve_and_repair_is_bounded(case):
    config, names, task = case
    task.consume(proposed_question(names))
    snapshot = deepcopy((task.values, task.next_item_id, task.pending_proposal, task.pending_clarification))
    for _ in range(config['demo']['max_repairs'] + 1):
        action = task.consume(response(is_affirmation=True))
        assert (task.values, task.next_item_id, task.pending_proposal, task.pending_clarification) == snapshot
        assert not task.confirmed
    assert action.kind == 'handoff'


def test_collection_schema_rejects_reserved_identity_and_overlapping_field_storage(case):
    config, names, _ = case
    for invalid in ['reserved_id', 'overlap']:
        changed = deepcopy(config)
        if invalid == 'reserved_id':
            next(slot for slot in changed['slots'] if slot['id'] == names['asset'])['id'] = 'id'
            changed['demo']['transaction_schema']['item_slots'][0] = 'id'
        else:
            changed['demo']['transaction_schema']['root_slots'].append(names['asset'])
        with pytest.raises(ValueError):
            validate_collection_schema(changed)
