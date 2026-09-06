"""Only exact current linked answers bypass inference; no general language parser."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from collection_fixtures import collection_config
from engine.collection_task import ConfiguredCollectionState
from engine.router import Router
from engine.scoped_answers import exact_linked_answer


def op(item_id, slot, value):
    return dict(op='set', item_id=item_id, slot=slot, value=value)


def response(**changes):
    value = dict(intent='task', ops=[], confidence=.99, unclear=False,
        is_affirmation=False, is_negation=False, clarification=None, proposed_ops=[],
        resolves_clarification=None, discard_clarification=None, discard_request=None)
    value.update(changes)
    return value


@pytest.fixture(params=[False, True], ids=['equipment', 'renamed'])
def case(request, monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    config, names = collection_config(renamed=request.param)
    config['router']['retries'] = 0
    asset = next(slot for slot in config['slots'] if slot['id'] == names['asset'])
    asset['aliases']['tripod'] += ['camera stand', 'حامل']
    task = ConfiguredCollectionState(config)
    task.apply([dict(op='create', item_id=1, slot=None, value=None), op(1, names['asset'], 'camera'),
        op(1, names['quantity'], 1), op(None, names['date'], '2026-09-15')])
    task.consume(response(intent='ambiguous', clarification=dict(kind='ambiguous_value',
        slot=names['asset'], item_ids=[1], linked_addresses=[{'item_id': 1, 'slot': names['quantity']}])))
    return config, names, task


@pytest.mark.parametrize('text', ['tripod', '  TRIPOD  ', 'Camera Stand', 'حامل'])
def test_exact_whole_configured_alias_is_a_single_scoped_set_without_mutation(case, text):
    config, names, task = case
    context = task.router_context()
    before = deepcopy(context)
    assert exact_linked_answer(config, context, text) == op(1, names['asset'], 'tripod')
    assert context == before and task.values[names['collection']][0][names['asset']] == 'camera'


@pytest.mark.parametrize('text', ['tripod please', 'choose tripod', 'tripod.', 'tripod?',
    'not tripod', 'tripod quantity three', 'tripod or camera', 'triPodium', 'trpod', 'yes', 'no', ''])
def test_nonexact_or_multi_fact_answer_abstains(case, text):
    config, _, task = case
    assert exact_linked_answer(config, task.router_context(), text) is None


def test_alias_collision_and_confirmation_aliases_do_not_choose_a_value(case):
    config, names, task = case
    asset = next(slot for slot in config['slots'] if slot['id'] == names['asset'])
    asset['aliases']['camera'].append('camera stand')
    assert exact_linked_answer(config, task.router_context(), 'camera stand') is None
    asset['aliases']['camera'].append('yes')
    asset['aliases']['tripod'].append('no')
    assert exact_linked_answer(config, task.router_context(), 'yes') is None
    assert exact_linked_answer(config, task.router_context(), 'no') is None


@pytest.mark.parametrize('text,expected', [('3', 3), (' 10 ', 10), ('0', None), ('11', None),
    ('3.0', None), ('+3', None), ('-3', None), ('٣', None), ('three', None), ('3 units', None)])
def test_current_integer_answer_is_ascii_only_and_bound_checked(case, text, expected):
    config, names, task = case
    task.consume(response(ops=[op(1, names['asset'], 'tripod')], resolves_clarification=task.pending_clarification['id']))
    before = deepcopy(task.router_context())
    result = exact_linked_answer(config, task.router_context(), text)
    assert result == (None if expected is None else op(1, names['quantity'], expected))
    assert task.router_context() == before


@pytest.mark.parametrize('invalid', ['no_question', 'no_proposal', 'unlinked', 'wrong_kind',
    'wrong_scope', 'wrong_row', 'boolean_row', 'stale_version', 'missing_version', 'already_answered'])
def test_missing_or_inconsistent_pending_authority_abstains(case, invalid):
    config, names, task = case
    context = task.router_context()
    if invalid == 'no_question':
        context['pending_clarification'] = None
    elif invalid == 'no_proposal':
        context['pending_proposal'] = None
    elif invalid == 'unlinked':
        context['pending_proposal'].pop('linked_addresses')
    elif invalid == 'wrong_kind':
        context['pending_clarification']['kind'] = 'unsupported_value'
    elif invalid == 'wrong_scope':
        context['pending_clarification']['slot'] = names['date']
    elif invalid == 'wrong_row':
        context['pending_clarification']['item_ids'] = [99]
    elif invalid == 'boolean_row':
        context['pending_clarification']['item_ids'] = [True]
    elif invalid == 'stale_version':
        context['pending_proposal']['base_version'] += 1
    elif invalid == 'missing_version':
        context.pop('version')
    else:
        context['pending_proposal']['remaining_addresses'] = [{'item_id': 1, 'slot': names['quantity']}]
    assert exact_linked_answer(config, context, 'tripod') is None


def test_noncollection_adapter_does_not_use_linked_fast_path(case):
    config, _, task = case
    for kind in ('flat_scoped', 'collection_scoped'):
        altered = deepcopy(config)
        altered['demo']['state_kind'] = kind
        assert exact_linked_answer(altered, task.router_context(), 'tripod') is None


def test_real_router_uses_no_http_or_quota_for_two_exact_answers_then_commits_unconfirmed(case, tmp_path):
    async def run():
        config, names, task = case
        def forbidden(request):
            pytest.fail('Exact linked answer contacted a provider')
        async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
            router = Router(config, tmp_path, api_key='offline-fixture', client=client)
            router.limiter.reserve = lambda: pytest.fail('Exact linked answer reserved provider quota')
            old = deepcopy(task.values)
            first_id = task.pending_clarification['id']
            first = await router.route('TRIPOD', task.router_context())
            assert first.resolves_clarification == first_id and first._routing_source == 'configured_exact_answer'
            assert [operation.model_dump() for operation in first.ops] == [op(1, names['asset'], 'tripod')]
            assert task.values == old
            task.consume(first.model_dump())
            assert task.values == old and task.pending_clarification['id'] > first_id
            second = await router.route('3', task.router_context())
            assert second.resolves_clarification == task.pending_clarification['id']
            assert task.consume(second.model_dump()).kind == 'readback'
            assert task.values[names['collection']][0] == {'id': 1, names['asset']: 'tripod', names['quantity']: 3}
            assert task.pending_proposal is None and task.readback_version is None and not task.confirmed
            assert router.calls == [] and len(router.local_calls) == 2
            assert all(call['source'] == 'configured_exact_answer' for call in router.local_calls)
    asyncio.run(run())


def test_nonexact_answer_uses_normal_provider_and_identity_guard(case, tmp_path):
    async def run():
        config, names, task = case
        calls, reserves = [], []
        value = response(ops=[op(1, names['asset'], 'tripod')], resolves_clarification=task.pending_clarification['id'])
        def reply(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                'message': {'content': json.dumps(value)}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
            router = Router(config, tmp_path, api_key='offline-fixture', client=client)
            router.limiter.reserve = lambda: reserves.append(True)
            output = await router.route('tripod please', task.router_context())
            assert len(calls) == len(reserves) == len(router.calls) == 1
            assert router.calls[0]['ok'] and router.local_calls == []
            assert output.resolves_clarification == task.pending_clarification['id']
    asyncio.run(run())
