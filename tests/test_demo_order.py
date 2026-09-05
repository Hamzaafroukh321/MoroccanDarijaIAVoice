"""Deterministic order semantics; these fixtures do not measure Darija accuracy."""

from copy import deepcopy
import json

import pytest

from engine.config import ROOT, load_config
from engine.demo_order import DemoOrderState


@pytest.fixture
def state():
    config = load_config(ROOT / 'configs/pizza.json')
    config['demo'] = json.loads((ROOT/'configs/demo/voice.json').read_text(encoding='utf-8'))
    return DemoOrderState(config)


def op(kind, item_id=None, slot=None, value=None):
    return {'op': kind, 'item_id': item_id, 'slot': slot, 'value': value}


def item(item_id, size='large', toppings=None):
    return [op('create', item_id), op('set', item_id, 'quantity', 1),
            op('set', item_id, 'size', size), op('set', item_id, 'toppings', ['cheese'] if toppings is None else toppings)]


def response(ops=None, *, yes=False, no=False, intent='task'):
    return {'ops': ops or [], 'is_affirmation': yes, 'is_negation': no,
            'unclear': intent != 'task', 'intent': intent}


def complete(state):
    state.apply(item(1) + [op('set', None, 'drink', ['cola'])])


def test_distinct_pizzas_keep_separate_fields_and_confirmation(state):
    state.consume(response(item(1) + item(2, 'medium', ['beef']) + [op('set', None, 'drink', ['cola'])]))
    assert state.values['items'] == [
        {'id': 1, 'quantity': 1, 'size': 'large', 'toppings': ['cheese']},
        {'id': 2, 'quantity': 1, 'size': 'medium', 'toppings': ['beef']}]
    assert state.ready and not state.confirmed
    assert state.consume(response(yes=True)).kind == 'readback'
    state.begin_confirmation(state.version)
    assert state.consume(response(yes=True)).kind == 'accepted'


def test_sequential_add_and_update_second_only(state):
    complete(state)
    first = deepcopy(state.values['items'][0])
    state.consume(response(item(2, 'medium', ['beef'])))
    state.consume(response([op('set', 2, 'size', 'small'), op('add', 2, 'toppings', 'olive')]))
    assert state.values['items'][0] == first
    assert state.values['items'][1]['size'] == 'small'
    assert state.values['items'][1]['toppings'] == ['beef', 'olive']


def test_delete_preserves_ids_and_never_reuses_deleted_ids(state):
    state.apply(item(1) + item(2))
    state.apply([op('delete', 1)] + item(3, 'small'))
    assert [value['id'] for value in state.values['items']] == [2, 3]
    before = deepcopy(state.values)
    with pytest.raises(ValueError):
        state.apply([op('create', 1)])
    assert state.values == before and state.next_item_id == 4
    state.apply([op('delete', 2), op('delete', 3)])
    assert state.next_action().item_id == 4 and not state.ready


def test_invalid_unknown_id_batch_is_atomic_including_allocator(state):
    complete(state)
    state.begin_confirmation(state.version)
    before = deepcopy(state.values)
    version = state.version
    with pytest.raises(ValueError):
        state.apply([op('create', 2), op('set', 1, 'size', 'small'), op('set', 3, 'size', 'large')])
    assert state.values == before and state.next_item_id == 2
    assert state.version == version and state.readback_version == version


@pytest.mark.parametrize('bad', [op('set', None, 'size', 'small'), op('set', 1, 'drink', ['cola']),
    op('create', 3), op('create', True), op('delete', 8), op('set', 1, 'quantity', True),
    op('set', 1, 'quantity', 21), op('set', 1, 'size', 'giant')])
def test_invalid_operations_never_mutate(state, bad):
    before = deepcopy(state.values)
    with pytest.raises(ValueError):
        state.apply([op('create', 1), bad])
    assert state.values == before and state.next_item_id == 1


def test_item_limit_is_atomic(state):
    state.config['demo']['order_max_items'] = 1
    with pytest.raises(ValueError):
        state.apply([op('create', 1), op('create', 2)])
    assert state.values == {'items': []} and state.next_item_id == 1


def test_prompt_does_not_create_or_infer_quantity_and_asks_sequentially(state):
    action = state.next_action()
    assert (action.kind, action.slot, action.item_id) == ('ask', 'size', 1)
    assert state.values == {'items': []} and state.version == 0
    state.apply([op('create', 1), op('set', 1, 'size', 'large'), op('create', 2)])
    assert (state.next_action().slot, state.next_action().item_id) == ('toppings', 1)
    state.apply([op('set', 1, 'toppings', [])])
    assert (state.next_action().slot, state.next_action().item_id) == ('quantity', 1)
    assert 'quantity' not in state.values['items'][0]


def test_plain_pizza_is_complete_but_absent_toppings_is_unknown(state):
    state.apply(item(1, toppings=[]) + [op('set', None, 'drink', ['none'])])
    assert state.ready
    state.apply([op('clear', 1, 'toppings')])
    assert not state.ready and state.next_action().slot == 'toppings'


def test_none_and_drink_are_exclusive_atomically(state):
    complete(state)
    before = deepcopy(state.values)
    with pytest.raises(ValueError):
        state.apply([op('set', 1, 'size', 'small'), op('add', None, 'drink', 'none')])
    assert state.values == before
    state.apply([op('set', None, 'drink', ['none'])])
    state.apply([op('set', None, 'drink', ['cola'])])
    assert state.values['drink'] == ['cola']


def test_no_and_changed_confirmation_need_new_readback(state):
    complete(state)
    state.begin_confirmation(state.version)
    assert state.consume(response(no=True)).kind == 'repair'
    assert state.awaiting_correction and state.readback_version is None
    assert state.consume(response(yes=True)).kind == 'repair'
    action = state.consume(response([op('set', 1, 'size', 'small')], yes=True))
    assert action.kind == 'readback' and not state.confirmed
    state.begin_confirmation(state.version - 1)
    assert state.readback_version is None
    state.begin_confirmation(state.version)
    assert state.consume(response(yes=True)).kind == 'accepted'


def test_redundant_correction_plus_yes_requires_fresh_readback(state):
    complete(state)
    state.begin_confirmation(state.version)
    assert state.consume(response([op('set', 1, 'size', 'large')], yes=True)).kind == 'readback'
    assert not state.confirmed and state.readback_version is None


def test_ambiguous_has_no_mutation_and_invalidates_confirmation(state):
    complete(state)
    state.begin_confirmation(state.version)
    before = deepcopy(state.values)
    assert state.consume(response(intent='ambiguous')).kind == 'ambiguous'
    assert state.values == before and state.readback_version is None
    assert state.awaiting_correction
    assert state.consume(response(yes=True)).kind == 'ambiguous'
    with pytest.raises(ValueError):
        state.consume(response([op('set', 1, 'size', 'small')], intent='ambiguous'))
    assert state.values == before


def test_ambiguity_blocks_stale_readback_and_redundant_change_until_real_correction(state):
    complete(state)
    state.apply(item(2, 'medium', ['beef']))
    original_version = state.version
    state.begin_confirmation(original_version)
    state.consume(response(intent='ambiguous'))
    state.begin_confirmation(original_version)
    assert state.readback_version is None and state.awaiting_correction
    assert state.consume(response([op('set', 1, 'size', 'large')], yes=True)).kind == 'ambiguous'
    assert state.awaiting_correction and not state.confirmed
    assert state.consume(response([op('set', 2, 'size', 'small')], yes=True)).kind == 'readback'
    assert not state.awaiting_correction and not state.ambiguity_pending
    assert not state.confirmed and state.readback_version is None
    state.begin_confirmation(state.version)
    assert state.consume(response(yes=True)).kind == 'accepted'


def test_ambiguous_then_repeated_yes_hands_off_without_confirmation(state):
    complete(state)
    state.begin_confirmation(state.version)
    state.consume(response(intent='ambiguous'))
    for _ in range(2):
        assert state.consume(response(yes=True)).kind == 'ambiguous'
    assert state.consume(response(yes=True)).kind == 'handoff'
    assert state.awaiting_correction and not state.confirmed


def test_repairs_bounded_and_only_progress_resets_count(state):
    state.apply([op('create', 1), op('set', 1, 'size', 'large')])
    for count in range(1, 4):
        action = state.consume(response(intent='unclear'))
        assert (action.kind, action.slot, action.item_id) == ('repair', 'toppings', 1)
        state.consume(response([op('set', 1, 'size', 'large')]))
        assert state.repair_count == count
    assert state.consume(response(intent='unclear')).kind == 'handoff'
    state.consume(response([op('set', 1, 'toppings', [])]))
    assert state.repair_count == 0


def test_greeting_does_not_consume_repair_budget_and_context_is_copy(state):
    for _ in range(5):
        state.consume(response(intent='greeting'))
    assert state.repair_count == 0
    context = state.router_context()
    assert context['next_item_id'] == context['requested_item_id'] == 1
    assert context['requested_slot'] == 'size'
    context['state']['items'].append({'id': 99})
    assert state.values == {'items': []}


def test_remove_last_topping_keeps_explicit_plain_choice(state):
    complete(state)
    state.begin_confirmation(state.version)
    state.apply([op('remove', 1, 'toppings', 'cheese')])
    assert state.values['items'][0]['toppings'] == [] and state.ready
    assert state.readback_version is None and not state.confirmed


def test_create_delete_in_one_batch_consumes_id_without_faking_progress(state):
    state.repair_count = 2
    state.apply([op('create', 1), op('delete', 1)])
    assert state.values == {'items': []} and state.next_item_id == 2
    assert state.repair_count == 2


def test_repeated_negation_has_bounded_spoken_repair(state):
    complete(state)
    state.begin_confirmation(state.version)
    for _ in range(3):
        assert state.consume(response(no=True)).kind == 'repair'
    assert state.consume(response(no=True)).kind == 'handoff'
    assert not state.confirmed


def test_remove_from_unknown_toppings_does_not_invent_plain_pizza(state):
    state.apply(item(1) + [op('clear', 1, 'toppings'), op('set', None, 'drink', ['cola'])])
    before = deepcopy(state.values)
    version = state.version
    state.apply([op('remove', 1, 'toppings', 'cheese')])
    assert state.values == before and state.version == version
    assert 'toppings' not in state.values['items'][0]
    assert not state.ready and state.next_action().slot == 'toppings'


def test_unrelated_drink_or_new_item_does_not_resolve_existing_item_ambiguity(state):
    complete(state)
    state.consume(response(intent='ambiguous'))
    state.apply([op('set', None, 'drink', ['water'])])
    assert state.awaiting_correction and state.ambiguity_pending and state.repair_count == 1
    state.apply(item(2, 'medium', ['beef']))
    assert state.awaiting_correction and state.ambiguity_pending and state.repair_count == 1
    state.begin_confirmation(state.version)
    assert state.readback_version is None
    assert state.consume(response(yes=True)).kind == 'ambiguous'
    state.apply([op('set', 1, 'size', 'small')])
    assert not state.awaiting_correction and not state.ambiguity_pending
    assert state.repair_count == 0


def test_deleting_existing_ambiguous_item_resolves_but_needs_fresh_readback(state):
    complete(state)
    state.apply(item(2))
    state.consume(response(intent='ambiguous'))
    state.apply([op('delete', 2)])
    assert not state.ambiguity_pending and not state.confirmed
    assert state.consume(response(yes=True)).kind == 'readback'


def test_ambiguity_with_no_items_resolves_when_first_item_is_created(state):
    state.consume(response(intent='ambiguous'))
    state.apply([op('set', None, 'drink', ['cola'])])
    assert state.ambiguity_pending
    state.apply(item(1))
    assert not state.ambiguity_pending and not state.awaiting_correction
    assert state.ready and not state.confirmed


def scoped(kind, slot=None, ids=None):
    return {**response(intent='ambiguous'), 'clarification': {'kind': kind, 'slot': slot, 'item_ids': ids or []},
            'resolves_clarification': None}


def resolve(clarification_id, operations, **flags):
    return {**response(operations, **flags), 'clarification': None, 'resolves_clarification': clarification_id}


def test_scoped_drink_repair_requires_matching_explicit_choice_and_new_readback(state):
    complete(state)
    state.begin_confirmation(state.version)
    before = deepcopy(state.values)
    assert state.consume(scoped('drink_size', 'drink')).kind == 'drink_size'
    assert state.values == before and not state.confirmed and state.readback_version is None
    pending_id = state.pending_clarification['id']
    assert state.consume(response([op('set', 1, 'size', 'small')])).kind == 'drink_size'
    assert state.pending_clarification['id'] == pending_id
    action = state.consume(resolve(pending_id, [op('set', None, 'drink', ['water'])], yes=True))
    assert action.kind == 'readback' and state.pending_clarification is None
    assert not state.confirmed and state.readback_version is None
    state.begin_confirmation(state.version)
    assert state.consume(response(yes=True)).kind == 'accepted'


def test_scoped_repair_cannot_be_cleared_by_compatible_change_without_resolution_id(state):
    complete(state)
    state.consume(scoped('unsupported_drink', 'drink'))
    pending = deepcopy(state.pending_clarification)
    before = deepcopy(state.values), state.version
    with pytest.raises(ValueError, match='resolution ID'):
        state.consume(response([op('set', None, 'drink', ['water'])]))
    assert (state.values, state.version) == before
    assert state.pending_clarification == pending and state.awaiting_correction
    state.begin_confirmation(state.version)
    assert state.readback_version is None


def test_scoped_bare_yes_no_do_not_resolve_and_failed_turns_are_bounded(state):
    complete(state)
    state.consume(scoped('drink_size', 'drink'))
    assert state.consume(response(yes=True)).kind == 'drink_size'
    assert state.consume(response(no=True)).kind == 'drink_size'
    assert state.consume(response(yes=True)).kind == 'handoff'
    assert state.pending_clarification is not None and not state.confirmed


def test_wrong_stale_resolution_id_rejects_batch_before_any_mutation(state):
    complete(state)
    state.consume(scoped('unsupported_drink', 'drink'))
    first_id = state.pending_clarification['id']
    state.consume(resolve(first_id, [op('set', None, 'drink', ['water'])]))
    state.consume(scoped('drink_size', 'drink'))
    assert state.pending_clarification['id'] > first_id
    before = deepcopy(state.values)
    version = state.version
    with pytest.raises(ValueError, match='stale'):
        state.consume(resolve(first_id, [op('create', 2), op('set', None, 'drink', ['water'])]))
    assert state.values == before and state.version == version and state.next_item_id == 2


def test_redundant_scoped_choice_advances_confirmation_epoch(state):
    complete(state)
    previous_version = state.version
    state.begin_confirmation(previous_version)
    state.consume(scoped('drink_size', 'drink'))
    pending_id = state.pending_clarification['id']
    assert state.consume(resolve(pending_id, [op('set', None, 'drink', ['cola'])], yes=True)).kind == 'readback'
    assert state.version > previous_version and state.readback_version is None
    state.begin_confirmation(previous_version)
    assert state.readback_version is None
    assert state.consume(response(yes=True)).kind == 'readback'
    state.begin_confirmation(state.version)
    assert state.consume(response(yes=True)).kind == 'accepted'


def test_item_reference_resolution_must_match_original_id_and_slot(state):
    complete(state)
    state.apply(item(2, 'medium', ['beef']))
    state.consume(scoped('item_reference', 'size', [2]))
    pending_id = state.pending_clarification['id']
    for bad in ([op('set', 1, 'size', 'small')], [op('set', 2, 'toppings', ['olive'])], item(3)):
        before = deepcopy(state.values)
        with pytest.raises(ValueError, match='scope'):
            state.consume(resolve(pending_id, bad))
        assert state.values == before
    assert state.consume(resolve(pending_id, [op('set', 2, 'size', 'small')])).kind == 'readback'
    assert state.values['items'][0]['size'] == 'large'


def test_empty_item_reference_ids_capture_existing_items_only_and_context_is_copy(state):
    complete(state)
    state.apply(item(2))
    assert state.consume(scoped('item_reference')).kind == 'item_reference'
    assert state.pending_clarification['item_ids'] == [1, 2]
    pending_id = state.pending_clarification['id']
    context = state.router_context()
    context['pending_clarification']['item_ids'].append(99)
    assert state.pending_clarification['item_ids'] == [1, 2]
    state.apply(item(3))
    with pytest.raises(ValueError, match='scope'):
        state.consume(resolve(pending_id, [op('delete', 3)]))
    assert state.consume(resolve(pending_id, [op('delete', 2)])).kind == 'readback'


@pytest.mark.parametrize('ids', [[9], [True], [-1]])
def test_invalid_clarification_ids_do_not_change_state_or_pending(state, ids):
    complete(state)
    state.begin_confirmation(state.version)
    before = deepcopy(state.values)
    with pytest.raises(ValueError):
        state.consume(scoped('item_reference', 'size', ids))
    assert state.values == before and state.pending_clarification is None
    assert state.readback_version == state.version


def test_resolution_with_invalid_late_operation_keeps_pending_and_batch_atomic(state):
    complete(state)
    state.consume(scoped('unsupported_drink', 'drink'))
    pending = deepcopy(state.pending_clarification)
    before = deepcopy(state.values)
    with pytest.raises(ValueError):
        state.consume(resolve(pending['id'], [op('set', None, 'drink', ['water']), op('set', 9, 'size', 'small')]))
    assert state.values == before and state.pending_clarification == pending


def test_resolution_with_no_operations_cannot_clear_pending(state):
    complete(state)
    state.consume(scoped('drink_size', 'drink'))
    with pytest.raises(ValueError, match='scope'):
        state.consume(resolve(state.pending_clarification['id'], [], yes=True))
    assert state.pending_clarification is not None and not state.confirmed


@pytest.mark.parametrize('kind,slot,operations', [
    ('order_details', None, item(2)),
    ('unsupported_menu', 'toppings', [op('set', 1, 'toppings', ['olive'])]),
    ('unsupported_menu', None, [op('set', None, 'drink', ['water'])]),
    ('unintelligible', None, [op('set', 1, 'quantity', 2)]),
])
def test_other_scoped_clarifications_resolve_with_compatible_operations(state, kind, slot, operations):
    complete(state)
    state.consume(scoped(kind, slot))
    pending_id = state.pending_clarification['id']
    assert state.consume(resolve(pending_id, operations)).kind == 'readback'
    assert state.pending_clarification is None and not state.confirmed


def test_scoped_explicit_resolution_clears_legacy_ambiguity_latch(state):
    complete(state)
    state.consume(response(intent='ambiguous'))
    assert state.ambiguity_pending
    state.consume(scoped('item_reference', 'size', [1]))
    pending_id = state.pending_clarification['id']
    state.consume(resolve(pending_id, [op('set', 1, 'size', 'large')]))
    assert not state.ambiguity_pending and not state.awaiting_correction
    assert state.pending_clarification is None and state.readback_version is None


def test_new_drink_clarification_cannot_replace_unresolved_pizza_reference(state):
    complete(state)
    state.apply(item(2, 'medium', ['beef']))
    state.consume(scoped('item_reference', 'size', [2]))
    original = deepcopy(state.pending_clarification)
    before = deepcopy(state.values)
    assert state.consume(scoped('unsupported_drink', 'drink')).kind == 'item_reference'
    assert state.pending_clarification == original and state.values == before
    assert state.repair_count == 2
    with pytest.raises(ValueError, match='scope'):
        state.consume(resolve(original['id'], [op('set', None, 'drink', ['water'])]))
    assert state.values == before and state.pending_clarification == original
    # Unrelated legitimate progress may be saved, but still cannot release the
    # original question or confirm its unresolved pizza reference.
    assert state.consume(response([op('set', None, 'drink', ['water'])])).kind == 'item_reference'
    assert state.values['drink'] == ['water']
    assert state.values['items'] == before['items'] and state.pending_clarification == original
    assert state.consume(resolve(original['id'], [op('set', 2, 'size', 'small')])).kind == 'readback'
    assert state.pending_clarification is None and not state.confirmed


def test_replacement_candidate_is_validated_without_losing_original_pending(state):
    complete(state)
    state.consume(scoped('item_reference', 'size', [1]))
    original = deepcopy(state.pending_clarification)
    with pytest.raises(ValueError, match='unknown pizza'):
        state.consume(scoped('item_reference', 'size', [99]))
    assert state.pending_clarification == original and state.repair_count == 1


def test_pending_actions_preserve_scoped_slot_and_single_item_id(state):
    complete(state)
    state.apply(item(2, 'medium', ['beef']))
    action = state.consume(scoped('unsupported_menu', 'toppings', [2]))
    assert (action.kind, action.slot, action.item_id) == ('unsupported_menu', 'toppings', 2)
    assert state.next_action() == action
    assert state.consume(response(yes=True)) == action


def test_multiple_scoped_items_are_not_reduced_to_an_arbitrary_item(state):
    complete(state)
    state.apply(item(2))
    action = state.consume(scoped('item_reference', 'size', [1, 2]))
    assert (action.kind, action.slot, action.item_id) == ('item_reference', 'size', None)
    assert state.next_action() == action


def test_unintelligible_without_slot_focuses_current_missing_item_field(state):
    state.apply([op('create', 1), op('set', 1, 'size', 'large')])
    action = state.consume(scoped('unintelligible'))
    assert (action.kind, action.slot, action.item_id) == ('unintelligible', 'toppings', 1)
    assert state.next_action() == action
    # A scoped answer must resolve the question before moving to the next gap.
    with pytest.raises(ValueError, match='resolution ID'):
        state.consume(response([op('set', 1, 'toppings', [])]))
    assert state.next_action() == action
    action = state.consume(resolve(state.pending_clarification['id'],
                                   [op('set', 1, 'toppings', [])]))
    assert (action.kind, action.slot, action.item_id) == ('ask', 'quantity', 1)
    assert state.pending_clarification is None


def test_unintelligible_preserves_explicit_slot_instead_of_missing_field(state):
    state.apply([op('create', 1)])
    action = state.consume(scoped('unintelligible', 'drink'))
    assert (action.kind, action.slot, action.item_id) == ('unintelligible', 'drink', None)


@pytest.mark.parametrize('kind', ['unsupported_menu', 'unintelligible', 'order_details'])
def test_all_clarification_kinds_enforce_explicit_item_and_slot_scope(state, kind):
    complete(state)
    state.apply(item(2, 'medium', ['beef']))
    state.consume(scoped(kind, 'toppings', [1]))
    pending_id = state.pending_clarification['id']
    before = deepcopy(state.values)
    for incompatible in ([op('set', 2, 'toppings', ['olive'])],
                         [op('set', None, 'drink', ['water'])],
                         [op('set', 1, 'size', 'small')]):
        with pytest.raises(ValueError, match='scope'):
            state.consume(resolve(pending_id, incompatible))
        assert state.values == before and state.pending_clarification['id'] == pending_id
    assert state.consume(resolve(pending_id, [op('set', 1, 'toppings', ['olive'])])).kind == 'readback'


def test_order_quantity_clarification_rejects_other_fields_and_other_items(state):
    complete(state)
    state.apply(item(2))
    state.consume(scoped('order_details', 'quantity', [1]))
    pending_id = state.pending_clarification['id']
    before = deepcopy(state.values)
    for incompatible in ([op('set', 1, 'size', 'small')],
                         [op('set', 2, 'quantity', 2)], item(3)):
        with pytest.raises(ValueError, match='scope'):
            state.consume(resolve(pending_id, incompatible))
        assert state.values == before
    assert state.consume(resolve(pending_id, [op('set', 1, 'quantity', 2)])).kind == 'readback'


@pytest.mark.parametrize('operations', [item(2), [op('delete', 1)]])
def test_order_wide_quantity_clarification_can_resolve_by_create_or_delete(state, operations):
    complete(state)
    state.consume(scoped('order_details', 'quantity'))
    pending_id = state.pending_clarification['id']
    state.consume(resolve(pending_id, operations))
    assert state.pending_clarification is None and not state.confirmed


def test_scoped_item_reference_can_resolve_by_deleting_target_despite_slot(state):
    complete(state)
    state.apply(item(2))
    state.consume(scoped('item_reference', 'size', [2]))
    pending_id = state.pending_clarification['id']
    assert state.consume(resolve(pending_id, [op('delete', 2)])).kind == 'readback'
    assert [item['id'] for item in state.values['items']] == [1]


def staged(operations, kind='unsupported_drink'):
    return {**scoped(kind, 'drink'), 'intent': 'out_of_scope', 'proposed_ops': operations,
            'discard_clarification': None}


def test_staged_two_pizzas_are_uncommitted_until_drink_resolution_and_apply_once(state):
    proposal_ops = item(1) + item(2, 'medium', ['beef'])
    assert state.consume(staged(proposal_ops)).kind == 'unsupported_drink'
    pending_id = state.pending_clarification['id']
    assert state.values == {'items': []} and state.version == 0 and state.next_item_id == 1
    assert not state.ready and not state.confirmed
    context = state.router_context()
    assert context['state'] == {'items': []}
    assert context['pending_proposal']['next_item_id'] == 3
    assert context['pending_proposal']['base_version'] == 0
    assert [row['size'] for row in context['pending_proposal']['state']['items']] == ['large', 'medium']
    context['pending_proposal']['state']['items'][0]['size'] = 'small'
    proposal_ops[1]['value'] = 20
    assert state.pending_proposal['state']['items'][0]['size'] == 'large'
    assert state.pending_proposal['ops'][1]['value'] == 1
    assert state.consume(resolve(pending_id, [op('set', None, 'drink', ['cola'])], yes=True)).kind == 'readback'
    assert len(state.values['items']) == 2 and state.next_item_id == 3
    assert state.pending_proposal is None and state.pending_clarification is None
    assert not state.confirmed and state.readback_version is None
    committed = deepcopy(state.values)
    with pytest.raises(ValueError, match='stale'):
        state.consume(resolve(pending_id, [op('set', None, 'drink', ['cola'])]))
    assert state.values == committed


def test_staged_resolution_can_correct_a_proposed_item_before_commit(state):
    state.consume(staged(item(1) + item(2, 'medium', ['beef']), 'drink_size'))
    pending_id = state.pending_clarification['id']
    assert state.consume(resolve(pending_id, [op('set', 2, 'size', 'small'),
            op('set', None, 'drink', ['water'])])).kind == 'readback'
    assert [row['size'] for row in state.values['items']] == ['large', 'small']
    assert state.values['drink'] == ['water']


@pytest.mark.parametrize('operations', [item(2), item(1) + [op('set', 8, 'size', 'small')],
    item(1) + [op('set', None, 'drink', ['cola'])], item(1) + [op('set', 1, 'quantity', 100)]])
def test_invalid_initial_proposal_rolls_back_all_state_and_ids(state, operations):
    with pytest.raises(ValueError):
        state.consume(staged(operations))
    assert state.values == {'items': []} and state.version == 0 and state.next_item_id == 1
    assert state.pending_clarification is None and state.pending_proposal is None
    assert state.repair_count == 0


def test_bad_combined_resolution_preserves_staged_proposal_and_committed_order(state):
    complete(state)
    before = deepcopy(state.values)
    state.consume(staged(item(2)))
    pending = deepcopy(state.pending_proposal)
    pending_id = state.pending_clarification['id']
    version = state.version
    with pytest.raises(ValueError):
        state.consume(resolve(pending_id, [op('set', None, 'drink', ['water']), op('set', 7, 'size', 'small')]))
    assert state.values == before and state.version == version and state.next_item_id == 2
    assert state.pending_proposal == pending and state.pending_clarification['id'] == pending_id


def test_staged_proposal_blocks_unrelated_edits_replacement_and_incompatible_resolution(state):
    complete(state)
    state.consume(staged(item(2)))
    pending_id = state.pending_clarification['id']
    before, pending = deepcopy(state.values), deepcopy(state.pending_proposal)
    with pytest.raises(ValueError, match='staged proposal'):
        state.consume(response([op('set', 1, 'size', 'small')]))
    with pytest.raises(ValueError, match='initial'):
        state.consume(staged(item(2, 'small')))
    with pytest.raises(ValueError, match='scope'):
        state.consume(resolve(pending_id, [op('set', 1, 'size', 'small')]))
    assert state.values == before and state.pending_proposal == pending
    assert state.consume(response(yes=True)).kind == 'unsupported_drink'
    assert state.pending_proposal == pending and not state.confirmed


def test_discard_staged_proposal_preserves_committed_state_and_allocator(state):
    complete(state)
    before, version, next_id = deepcopy(state.values), state.version, state.next_item_id
    state.begin_confirmation(version)
    state.consume(staged(item(2)))
    pending_id = state.pending_clarification['id']
    discard = {**response(), 'discard_clarification': pending_id}
    assert state.consume(discard).kind == 'readback'
    assert state.values == before and state.next_item_id == next_id
    assert state.version > version and state.pending_proposal is None and state.pending_clarification is None
    state.begin_confirmation(version)
    assert state.readback_version is None and not state.confirmed
    with pytest.raises(ValueError, match='stale'):
        state.consume(discard)
    # Reserved preview IDs were never committed, and can now be used normally.
    state.apply(item(2, 'small'))
    assert state.next_item_id == 3


def test_discard_empty_order_returns_missing_prompt_without_creating_items(state):
    state.consume(staged(item(1)))
    action = state.consume({**response(), 'discard_clarification': state.pending_clarification['id']})
    assert (action.kind, action.slot, action.item_id) == ('ask', 'size', 1)
    assert state.values == {'items': []} and state.next_item_id == 1


def test_invalid_discard_or_wrong_resolution_id_preserves_proposal(state):
    state.consume(staged(item(1)))
    pending_id = state.pending_clarification['id']
    before = deepcopy(state.pending_proposal)
    for bad in ({**response(yes=True), 'discard_clarification': pending_id},
                {**response(), 'discard_clarification': pending_id + 1},
                resolve(pending_id + 1, [op('set', None, 'drink', ['cola'])])):
        with pytest.raises(ValueError):
            state.consume(bad)
        assert state.pending_proposal == before and state.values == {'items': []}


def test_existing_pending_question_cannot_be_replaced_by_staged_pizzas(state):
    complete(state)
    state.consume(scoped('item_reference', 'size', [1]))
    pending = deepcopy(state.pending_clarification)
    with pytest.raises(ValueError, match='initial'):
        state.consume(staged(item(2)))
    assert state.pending_clarification == pending and state.pending_proposal is None
