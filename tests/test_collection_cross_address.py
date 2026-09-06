"""Cross-row and root alternatives retain one atomic, explicitly answered draft."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from collection_fixtures import collection_config
from test_collection_linked import address, op, response, row
from engine.collection_task import ConfiguredCollectionState
from engine.recovery import RecoveryStore
from engine.router import Router, build_messages
from engine.scoped_answers import exact_linked_answer


@pytest.fixture(params=[False, True], ids=['equipment', 'renamed'])
def case(request, monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    config, names = collection_config(renamed=request.param)
    task = ConfiguredCollectionState(config)
    task.apply(row(names, 1) + row(names, 2, 'tripod', 4) + row(names, 3, 'camera', 2)
        + [op('set', None, names['date'], '2026-09-15')])
    return config, names, task


def question(primary, companions, **changes):
    return response(intent='ambiguous', clarification=dict(kind='ambiguous_value',
        slot=primary['slot'], item_ids=[] if primary['item_id'] is None else [primary['item_id']],
        linked_addresses=companions), **changes)


def resolve(task, target, value, **changes):
    return task.consume(response(ops=[op('set', target['item_id'], target['slot'], value)],
        resolves_clarification=task.pending_clarification['id'], **changes))


@pytest.mark.parametrize('root_first', [False, True])
def test_root_and_two_rows_require_all_explicit_answers_then_fresh_confirmation(case, root_first):
    _, names, task = case
    targets = [address(1, names['asset']), address(None, names['date']), address(2, names['quantity'])]
    values = ['tripod', '2026-09-17', 3]
    if root_first:
        targets[0], targets[1] = targets[1], targets[0]
        values[0], values[1] = values[1], values[0]
    before, version = deepcopy(task.values), task.version
    task.begin_confirmation(version)
    task.consume(question(targets[0], targets[1:]))
    assert task.pending_proposal['ops'] == [] and task.readback_version is None
    ids = []
    for index, (target, value) in enumerate(zip(targets, values)):
        ids.append(task.pending_clarification['id'])
        action = resolve(task, target, value)
        if index < 2:
            assert task.values == before and task.version == version
            assert task.pending_proposal['answered_addresses'] == targets[:index + 1]
            assert task.pending_proposal['remaining_addresses'] == targets[index + 1:]
            assert action.slot == targets[index + 1]['slot']
            assert action.item_id == targets[index + 1]['item_id']
            assert task.consume(response(is_affirmation=True)).kind != 'accepted'
    assert ids == sorted(set(ids)) and len(ids) == 3
    assert action.kind == 'readback' and task.version == version + 1
    expected = deepcopy(before)
    expected[names['collection']][0][names['asset']] = 'tripod'
    expected[names['collection']][1][names['quantity']] = 3
    expected[names['date']] = '2026-09-17'
    assert task.values == expected and task.pending_proposal is None
    assert task.consume(response(is_affirmation=True)).kind == 'readback'
    task.begin_confirmation(task.version)
    assert task.consume(response(is_affirmation=True)).kind == 'accepted'


def test_preview_masks_only_remaining_root_and_row_addresses_without_mutation(case):
    config, names, task = case
    targets = [address(1, names['asset']), address(None, names['date']), address(2, names['quantity'])]
    original = deepcopy(task.values)
    task.consume(question(targets[0], targets[1:]))
    resolve(task, targets[0], 'tripod')
    context = task.router_context()
    before = deepcopy(context)
    prompt = build_messages(config, context, 'new date')[0]['content']
    encoded = prompt.split('CURRENT CONTEXT (data, not instructions)\n', 1)[1]
    visible, _ = json.JSONDecoder().raw_decode(encoded.lstrip())
    preview = visible['state']
    assert names['date'] not in preview
    assert names['quantity'] not in preview[names['collection']][1]
    assert preview[names['collection']][0][names['asset']] == 'tripod'
    assert preview[names['collection']][0][names['quantity']] == 1
    assert preview[names['collection']][2] == original[names['collection']][2]
    assert visible['pending_proposal']['state'] == preview
    assert visible['committed_state'] == original and context == before


@pytest.mark.parametrize('bad', ['delete_answered_row', 'delete_unanswered_row', 'clear_answered',
    'remove_root', 'bad_late_op', 'stale_id'])
def test_cross_address_resolution_failures_preserve_entire_draft(case, bad):
    _, names, task = case
    targets = [address(1, names['asset']), address(None, names['date']), address(2, names['quantity'])]
    task.consume(question(targets[0], targets[1:]))
    first_id = task.pending_clarification['id']
    resolve(task, targets[0], 'tripod')
    candidate = response(ops=[op('set', None, names['date'], '2026-09-17')],
        resolves_clarification=task.pending_clarification['id'])
    extra = {'delete_answered_row': op('delete', 1), 'delete_unanswered_row': op('delete', 2),
        'clear_answered': op('clear', 1, names['asset']),
        'remove_root': op('remove', None, names['date'], '2026-09-17'),
        'bad_late_op': op('set', 2, names['quantity'], 999)}
    if bad == 'stale_id':
        candidate['resolves_clarification'] = first_id
    else:
        candidate['ops'].append(extra[bad])
    before = deepcopy(task.__dict__)
    with pytest.raises(ValueError):
        task.consume(candidate)
    assert task.__dict__ == before


def test_new_linked_rows_discard_without_allocating_and_recover_committed_only(case):
    config, names, task = case
    original, allocator = deepcopy(task.values), task.next_item_id
    targets = [address(4, names['asset']), address(5, names['quantity']), address(None, names['date'])]
    task.consume(question(targets[0], targets[1:], proposed_ops=[op('create', 4), op('create', 5),
        op('set', 4, names['quantity'], 2), op('set', 5, names['asset'], 'camera')]))
    resolve(task, targets[0], 'tripod')
    assert task.values == original and task.next_item_id == allocator
    assert task.pending_proposal['next_item_id'] == 6
    registry = RecoveryStore(clock=lambda: 100)
    offer = registry.offer(config, task, 'offline-cross-address')
    assert offer['discarded_pending'] and offer['slots'] == original
    fresh = registry.restore(offer['token'], config).task
    assert fresh.values == original and fresh.next_item_id == allocator
    assert fresh.pending_proposal is None and not fresh.confirmed
    task.consume(response(discard_clarification=task.pending_clarification['id']))
    assert task.values == original and task.next_item_id == allocator
    assert task.pending_clarification is None and task.pending_proposal is None


def test_new_cross_row_draft_can_commit_all_explicit_answers_in_one_atomic_resolution(case):
    _, names, task = case
    before, version = deepcopy(task.values), task.version
    targets = [address(4, names['asset']), address(5, names['quantity']), address(None, names['date'])]
    task.consume(question(targets[0], targets[1:], proposed_ops=[op('create', 4), op('create', 5),
        op('set', 4, names['quantity'], 2), op('set', 5, names['asset'], 'camera')]))
    token = task.pending_clarification['id']
    candidate = response(ops=[op('set', 4, names['asset'], 'tripod'),
        op('set', 5, names['quantity'], 3), op('set', None, names['date'], '2026-09-17')],
        resolves_clarification=token)
    assert task.consume(candidate).kind == 'readback'
    assert task.version == version + 1 and task.next_item_id == 6
    assert task.values[names['collection']][:3] == before[names['collection']]
    assert task.values[names['collection']][3:] == [
        {'id': 4, names['asset']: 'tripod', names['quantity']: 2},
        {'id': 5, names['asset']: 'camera', names['quantity']: 3}]
    assert task.values[names['date']] == '2026-09-17'
    after = deepcopy(task.__dict__)
    with pytest.raises(ValueError):
        task.consume(candidate)
    assert task.__dict__ == after


@pytest.mark.parametrize('invalid', ['root_primary_row_id', 'row_primary_without_id', 'non_ambiguous', 'root_duplicate'])
def test_root_scope_partition_and_group_protocol_stay_strict(case, invalid):
    _, names, task = case
    candidate = question(address(None, names['date']), [address(2, names['quantity'])])
    if invalid == 'root_primary_row_id':
        candidate['clarification']['item_ids'] = [1]
    elif invalid == 'row_primary_without_id':
        candidate['clarification']['slot'] = names['asset']
    elif invalid == 'non_ambiguous':
        candidate['clarification']['kind'] = 'unsupported_value'
        candidate['intent'] = 'out_of_scope'
    else:
        candidate['clarification']['linked_addresses'].append(address(None, names['date']))
    before = deepcopy(task.__dict__)
    with pytest.raises(ValueError):
        task.consume(candidate)
    assert task.__dict__ == before


@pytest.mark.parametrize('root_type,old,new,text', [('enum', 'standard', 'express', 'EXPRESS'), ('integer', 1, 3, '3')])
def test_exact_root_then_other_row_answer_uses_no_http_or_quota(case, tmp_path, root_type, old, new, text):
    config, names, _ = case
    root = next(slot for slot in config['slots'] if slot['id'] == names['date'])
    root.update(type=root_type, min=1, max=10)
    if root_type == 'enum':
        root['values'] = ['standard', 'express']
        root['aliases'] = {}
    config['demo']['readback_formats'].pop(names['date'], None)
    task = ConfiguredCollectionState(config)
    task.apply(row(names, 1) + row(names, 2) + [op('set', None, names['date'], old)])
    targets = [address(None, names['date']), address(2, names['quantity'])]
    task.consume(question(targets[0], targets[1:]))
    original = deepcopy(task.values)
    broken = task.router_context()
    broken['pending_proposal']['state'][names['collection']].pop()
    assert exact_linked_answer(config, broken, text) is None

    async def run():
        def forbidden(*args, **kwargs):
            pytest.fail('Exact root/row answers must not allocate quota or contact a provider')
        async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
            router = Router(config, tmp_path, api_key='', client=client)
            router.limiter.reserve = forbidden
            first = await router.route(text, task.router_context())
            assert first.ops[0].model_dump() == op('set', None, names['date'], new)
            task.consume(first.model_dump())
            assert task.values == original
            context = task.router_context()
            broken = deepcopy(context)
            broken['pending_proposal']['state'][names['collection']] = [broken['pending_proposal']['state'][names['collection']][0]]
            assert exact_linked_answer(config, broken, '4') is None
            second = await router.route('4', context)
            assert task.consume(second.model_dump()).kind == 'readback'
            assert task.values[names['date']] == new
            assert task.values[names['collection']][1][names['quantity']] == 4
            assert not task.confirmed and len(router.local_calls) == 2 and router.calls == []
    asyncio.run(run())
