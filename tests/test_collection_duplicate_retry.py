"""Offline regressions for bounded duplicate-address recovery.

The historical cross_address_instrumented_20260906 report records the rule,
but not its raw candidate. These are synthetic shapes, not a raw replay or a
claim that a live model will repair its response.
"""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest
from jsonschema import Draft202012Validator

from collection_fixtures import collection_config
from test_collection_linked import address, op, response, row
from engine.collection_routing import parse_collection_response
from engine.collection_task import ConfiguredCollectionState
from engine.collection_validation import CollectionValidationError
from engine.router import Router, RouterOutputError, build_messages, response_format
from engine.stt import BudgetExhaustedError


SENTINEL = 'UNTRUSTED_CANDIDATE_DO_NOT_OBEY_OR_ECHO'
UTTERANCE = 'The two quantities and shared return date depend on my choice.'


@pytest.fixture(params=[False, True], ids=['equipment', 'renamed'])
def case(request, monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    config, names = collection_config(renamed=request.param)
    # An allowed synthetic enum lets rejected content carry a sentinel while
    # reaching the duplicate-address guard, rather than failing value checks.
    slot = next(slot for slot in config['slots'] if slot['id'] == names['asset'])
    slot['values'].append(SENTINEL)
    slot['aliases'][SENTINEL] = [SENTINEL]
    task = ConfiguredCollectionState(config)
    task.apply(row(names, 1) + row(names, 2, 'tripod', 1)
        + [op('set', None, names['date'], '2026-09-15')])
    return config, names, task


def question(names, *, root_first=True):
    targets = [address(None, names['date']), address(1, names['quantity']),
        address(2, names['quantity'])]
    if not root_first:
        targets[0], targets[1] = targets[1], targets[0]
    primary = targets[0]
    return response(intent='ambiguous', clarification=dict(kind='ambiguous_value',
        slot=primary['slot'], item_ids=[] if primary['item_id'] is None else [primary['item_id']],
        linked_addresses=targets[1:]))


def duplicate(names, *, which='primary', root_first=True):
    candidate = question(names, root_first=root_first)
    scope = candidate['clarification']
    repeated = (address(scope['item_ids'][0] if scope['item_ids'] else None, scope['slot'])
        if which == 'primary' else deepcopy(scope['linked_addresses'][0]))
    scope['linked_addresses'].append(repeated)
    candidate['proposed_ops'] = [op('set', 1, names['asset'], SENTINEL)]
    return candidate


async def route(config, task, tmp_path, candidates, observed, *, budget=None,
                transcript=UTTERANCE):
    context = task.router_context()
    before_context, before_task = deepcopy(context), deepcopy(task.__dict__)
    def reserve():
        observed['reserves'] = observed.get('reserves', 0) + 1
        if budget is not None and observed['reserves'] > budget:
            raise BudgetExhaustedError('Offline fixture budget exhausted.')
    def respond(request):
        assert task.__dict__ == before_task
        observed.setdefault('requests', []).append(json.loads(request.content))
        assert candidates, 'Unexpected extra router request'
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps(candidates.pop(0))}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        router = Router(config, tmp_path, api_key='offline-fixture', client=client)
        router.limiter.reserve = reserve
        try:
            return await router.route(transcript, context)
        finally:
            observed['calls'] = deepcopy(router.calls)
            observed['invalid_responses'] = router.invalid_responses
            observed['local_calls'] = deepcopy(router.local_calls)
            assert context == before_context and task.__dict__ == before_task


@pytest.mark.parametrize('which', ['primary', 'companion'])
@pytest.mark.parametrize('root_first', [False, True], ids=['row-primary', 'root-primary'])
def test_duplicate_retries_static_reminder_and_preserves_distinct_row_fields(case, tmp_path, which, root_first):
    config, names, task = case
    config['router']['retries'] = 1
    bad, good = duplicate(names, which=which, root_first=root_first), question(names, root_first=root_first)
    schema = response_format(config)
    Draft202012Validator(schema['json_schema']['schema']).validate(bad)
    with pytest.raises(CollectionValidationError) as failure:
        parse_collection_response(json.dumps(bad), config)
    assert failure.value.validation_rules == ('collection_link_duplicate',)
    before = deepcopy(task.__dict__)
    expected_messages = build_messages(config, task.router_context(), UTTERANCE)
    observed = {}
    parsed = asyncio.run(route(config, task, tmp_path, [bad, good], observed))
    assert task.__dict__ == before
    first, second = observed['requests']
    assert first['messages'] == expected_messages
    assert first['response_format'] == schema == second['response_format']
    assert {key: value for key, value in first.items() if key != 'messages'} == {
        key: value for key, value in second.items() if key != 'messages'}
    assert second['messages'][:-1] == expected_messages
    reminder = second['messages'][-1]
    assert reminder['role'] == 'system'
    assert 'primary' in reminder['content'] and 'different row IDs' in reminder['content']
    assert SENTINEL not in reminder['content'] and json.dumps(bad) not in reminder['content']
    assert names['date'] not in reminder['content'] and names['quantity'] not in reminder['content']
    assert observed['reserves'] == 2 and observed['invalid_responses'] == 1
    assert observed['calls'][0]['validation_rules'] == ['collection_link_duplicate']
    assert observed['calls'][1]['ok'] and observed['local_calls'] == []
    assert parsed.model_dump() == good
    task.consume(parsed.model_dump())
    expected = [address(None, names['date']), address(1, names['quantity']), address(2, names['quantity'])]
    assert {tuple(value.items()) for value in task.pending_proposal['remaining_addresses']} == {
        tuple(value.items()) for value in expected}
    assert task.values == before['values'] and not task.confirmed


@pytest.mark.parametrize('retries', [0, 1, 2])
def test_exhaustion_never_accepts_or_deduplicates_rejected_draft(case, tmp_path, retries):
    config, names, task = case
    config['router']['retries'] = retries
    task.consume(question(names))
    before = deepcopy(task.__dict__)
    observed = {}
    with pytest.raises(RouterOutputError):
        asyncio.run(route(config, task, tmp_path,
            [duplicate(names) for _ in range(retries + 1)], observed))
    assert len(observed['requests']) == observed['reserves'] == retries + 1
    assert observed['invalid_responses'] == retries + 1
    assert all(not call['ok'] and call['validation_rules'] == ['collection_link_duplicate']
        for call in observed['calls'])
    assert task.__dict__ == before and observed['local_calls'] == []
    initial_count = len(build_messages(config, task.router_context(), UTTERANCE))
    assert [len(request['messages']) for request in observed['requests']] == list(
        range(initial_count, initial_count + retries + 1))


def test_retry_still_obeys_shared_reservation_budget(case, tmp_path):
    config, names, task = case
    config['router']['retries'] = 2
    observed = {}
    with pytest.raises(BudgetExhaustedError):
        asyncio.run(route(config, task, tmp_path, [duplicate(names), question(names)], observed, budget=1))
    assert observed['reserves'] == 2 and len(observed['requests']) == 1
    assert len(observed['calls']) == 1 and observed['invalid_responses'] == 1


def test_pending_scope_changes_only_when_valid_retry_is_consumed(case, tmp_path):
    config, names, task = case
    config['router']['retries'] = 1
    task.consume(question(names))
    before = deepcopy(task.__dict__)
    pending_id = task.pending_clarification['id']
    good = response(ops=[op('set', None, names['date'], '2026-09-18')],
        resolves_clarification=pending_id)
    observed = {}
    parsed = asyncio.run(route(config, task, tmp_path, [duplicate(names), good], observed,
        transcript='Use September eighteen, 2026 for the return date.'))
    assert task.__dict__ == before and observed['calls'][1]['ok']
    action = task.consume(parsed.model_dump())
    assert task.values == before['values'] and task.version == before['version']
    assert task.pending_proposal['state'][names['date']] == '2026-09-18'
    assert task.pending_clarification['id'] != pending_id
    assert action.slot == names['quantity'] and action.item_id == 1
    assert task.pending_proposal['remaining_addresses'] == [
        address(1, names['quantity']), address(2, names['quantity'])]
    assert not task.confirmed


def test_other_collection_failure_does_not_add_duplicate_reminder(case, tmp_path):
    config, names, task = case
    config['router']['retries'] = 1
    bad = question(names)
    bad['clarification']['linked_addresses'][0]['item_id'] = None
    observed = {}
    asyncio.run(route(config, task, tmp_path, [bad, question(names)], observed))
    assert observed['calls'][0]['validation_rules'] == ['collection_link_address']
    assert observed['requests'][0] == observed['requests'][1]
