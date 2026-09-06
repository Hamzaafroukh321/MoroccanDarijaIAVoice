"""Exact configured flat linked answers share routing without interpreting time."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.router import Router, build_messages
from engine.scoped_answers import exact_linked_answer
from engine.scoped_task import ScopedTaskState


def op(slot, value):
    return dict(op='set', slot=slot, value=value)


def response(**changes):
    value = dict(intent='task', ops=[], confidence=.99, unclear=False,
        is_affirmation=False, is_negation=False, clarification=None, proposed_ops=[],
        resolves_clarification=None, discard_clarification=None, discard_request=None)
    value.update(changes)
    return value


@pytest.fixture(params=['clinic', 'counter'], ids=['clinic-time', 'synthetic-integer'])
def case(request, monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    config = demo_config(load_config(ROOT / 'configs/clinic.json'))
    config['router']['retries'] = 0
    field, companion, old_companion = 'doctor', 'time', '10:00'
    if request.param == 'counter':
        # Entirely synthetic config: enum and bounded integer, no new clinic aliases.
        config = json.loads(json.dumps(config).replace('doctor', 'counter').replace('"time"', '"units"'))
        config['domain_id'] = 'counter-diagnostic'
        field, companion, old_companion = 'counter', 'units', 1
        definition = next(slot for slot in config['slots'] if slot['id'] == companion)
        definition.update(type='integer', min=1, max=10)
        config['demo']['readback_formats'].pop(companion)
        config['demo']['questions'][companion] = 'How many units?'
        config['demo']['labels'][companion] = 'Units'
    task = ScopedTaskState(config)
    task.apply([op(field, field + '_a'), op(companion, old_companion), op('date', '2026-09-15')])
    task.consume(response(intent='ambiguous', clarification=dict(kind='ambiguous_value',
        slot=field, item_ids=[], coupled_slots=[companion])))
    return config, field, companion, task


def test_exact_existing_enum_alias_produces_only_three_key_operation(case):
    config, field, _, task = case
    context, before = task.router_context(), deepcopy(task.__dict__)
    for text in (field + '_b', field + ' B', '  ' + field.upper() + ' B  '):
        assert exact_linked_answer(config, context, text) == op(field, field + '_b')
    assert task.__dict__ == before
    assert set(exact_linked_answer(config, context, field + ' B')) == {'op', 'slot', 'value'}


def test_nonexact_confirmation_and_multiple_facts_stay_with_normal_router(case):
    config, field, _, task = case
    for text in ('yes', 'no', field + ' B?', field + ' B.', 'choose ' + field + ' B',
                 field + ' A or ' + field + ' B', field + ' B and tomorrow', 'unconfigured'):
        assert exact_linked_answer(config, task.router_context(), text) is None


@pytest.mark.parametrize('invalid', ['missing_question', 'missing_proposal', 'unlinked',
    'wrong_request', 'stale_version', 'missing_version', 'wrong_coverage', 'row_identity'])
def test_missing_or_inconsistent_flat_authority_cannot_use_exact_fast_path(case, invalid):
    config, field, companion, task = case
    context = task.router_context()
    if invalid == 'missing_question':
        context['pending_clarification'] = None
    elif invalid == 'missing_proposal':
        context['pending_proposal'] = None
    elif invalid == 'unlinked':
        context['pending_proposal'].pop('coupled_slots')
    elif invalid == 'wrong_request':
        context['requested_slot'] = companion
    elif invalid == 'stale_version':
        context['pending_proposal']['base_version'] += 1
    elif invalid == 'missing_version':
        context.pop('version')
    elif invalid == 'wrong_coverage':
        context['pending_proposal']['remaining_slots'] = [companion]
    else:
        context['pending_clarification']['item_ids'] = [1]
    before = deepcopy(context)
    assert exact_linked_answer(config, context, field + ' B') is None
    assert context == before


def test_only_integer_companion_is_exact_time_always_abstains(case):
    config, field, companion, task = case
    task.consume(response(ops=[op(field, field + '_b')], resolves_clarification=task.pending_clarification['id']))
    if companion == 'time':
        for text in ('11', '11:00', '11h00'):
            assert exact_linked_answer(config, task.router_context(), text) is None
    else:
        assert exact_linked_answer(config, task.router_context(), '3') == op(companion, 3)
        for text in ('11', '0', 'three', '٣', '3 units'):
            assert exact_linked_answer(config, task.router_context(), text) is None


def test_saved_asr_homophone_is_not_rewritten_into_existing_configured_alias(case):
    config, field, _, task = case
    assert exact_linked_answer(config, task.router_context(), 'الطبيب باء') == op(field, field + '_b')
    assert exact_linked_answer(config, task.router_context(), 'الطبيب باع') is None


def test_requested_date_never_uses_numeric_or_iso_fast_path(case):
    config, field, _, original = case
    task = ScopedTaskState(config)
    task.apply([op(key, value) for key, value in original.values.items()])
    task.consume(response(intent='ambiguous', clarification=dict(kind='ambiguous_value',
        slot='date', item_ids=[], coupled_slots=[field])))
    assert task.router_context()['requested_slot'] == 'date'
    for text in ('2026-09-15', '11', '11:00'):
        assert exact_linked_answer(config, task.router_context(), text) is None


def test_real_router_counts_exact_enum_locally_then_routes_time_or_integer_appropriately(case, tmp_path):
    async def run():
        config, field, companion, task = case
        requests, reserves = [], []
        def reply(request):
            requests.append(json.loads(request.content))
            assert companion == 'time', 'Integer companion must not contact a provider'
            answer = response(ops=[op('time', '11:00')], resolves_clarification=task.pending_clarification['id'])
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                'message': {'content': json.dumps(answer)}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
            router = Router(config, tmp_path, api_key='offline-fixture', client=client)
            router.limiter.reserve = lambda: reserves.append(True)
            original, version = deepcopy(task.values), task.version
            first_id = task.pending_clarification['id']
            first = await router.route(field + ' B', task.router_context())
            assert [item.model_dump() for item in first.ops] == [op(field, field + '_b')]
            assert first.resolves_clarification == first_id and first._routing_source == 'configured_exact_answer'
            assert requests == reserves == router.calls == [] and len(router.local_calls) == 1
            task.consume(first.model_dump())
            assert task.values == original and task.version == version
            assert task.pending_clarification['id'] > first_id and task.next_action().slot == companion
            answer = await router.route('11:00' if companion == 'time' else '3', task.router_context())
            assert answer.resolves_clarification == task.pending_clarification['id']
            assert task.consume(answer.model_dump()).kind == 'readback'
            assert task.values[field] == field + '_b'
            assert task.values[companion] == ('11:00' if companion == 'time' else 3)
            assert task.version == version + 1 and task.readback_version is None and not task.confirmed
            expected_provider = int(companion == 'time')
            assert len(requests) == len(reserves) == len(router.calls) == expected_provider
            assert len(router.local_calls) == 2 - expected_provider
            assert all(item['source'] == 'configured_exact_answer' for item in router.local_calls)
    asyncio.run(run())


def test_nonexact_flat_answer_falls_back_to_normal_http_without_local_telemetry(case, tmp_path):
    async def run():
        config, field, _, task = case
        requests, reserves = [], []
        answer = response(ops=[op(field, field + '_b')], resolves_clarification=task.pending_clarification['id'])
        def reply(request):
            requests.append(request)
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                'message': {'content': json.dumps(answer)}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
            router = Router(config, tmp_path, api_key='offline-fixture', client=client)
            router.limiter.reserve = lambda: reserves.append(True)
            result = await router.route('I choose ' + field + ' B', task.router_context())
            assert len(requests) == len(reserves) == len(router.calls) == 1
            assert router.local_calls == [] and result.resolves_clarification == task.pending_clarification['id']
    asyncio.run(run())


def test_every_router_visible_flat_preview_masks_unanswered_companion_without_state_mutation(case):
    config, field, companion, task = case
    committed = deepcopy(task.values)
    local_answer = exact_linked_answer(config, task.router_context(), field + ' B')
    assert local_answer == op(field, field + '_b')
    task.consume(response(ops=[local_answer], resolves_clarification=task.pending_clarification['id']))
    context = task.router_context()
    # The internal transaction still starts from old committed values; only the
    # detached router preview hides values awaiting a new explicit answer.
    assert context['pending_proposal']['state'][companion] == committed[companion]
    original_context, original_task = deepcopy(context), deepcopy(task.__dict__)
    messages = build_messages(config, context, 'answer the next field')
    encoded = messages[0]['content'].split('\nCONTEXT: ', 1)[1]
    visible, _ = json.JSONDecoder().raw_decode(encoded)
    assert visible['committed_state'] == committed
    assert visible['state'][field] == field + '_b'
    assert companion not in visible['state']
    assert visible['pending_proposal']['state'] == visible['state']
    assert visible['requested_slot'] == companion
    assert context == original_context and task.__dict__ == original_task
