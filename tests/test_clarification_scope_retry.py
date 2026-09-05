"""Missing flat clarification scopes get bounded feedback, never guessed state."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.demo_order import DemoOrderState
from engine.router import Router, RouterOutputError, parse_response
from engine.scoped_task import ScopedTaskState


def payload(kind='ambiguous_value', slot=None):
    return dict(intent='ambiguous' if kind in {'ambiguous_value', 'order_details'} else 'out_of_scope',
        ops=[], proposed_ops=[], confidence=.99, unclear=False, is_affirmation=False,
        is_negation=False, clarification={'kind': kind, 'slot': slot, 'item_ids': []},
        resolves_clarification=None, discard_clarification=None, discard_request=None)


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    result = demo_config(load_config(ROOT / 'configs/clinic.json'))
    result['router']['retries'] = 1
    return result


def take_snapshot(state):
    return deepcopy((state.values, state.version, state.pending_clarification,
        state.pending_proposal, state.readback_version, state.confirmed_version))


async def run_router(config, root, context, candidates):
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        candidate = candidates[min(len(requests) - 1, len(candidates) - 1)]
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps(candidate)}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        router = Router(config, root, api_key='offline', client=client)
        router.limiter.reserve = lambda: None
        original = deepcopy(context)
        try:
            result = await router.route('Doctor A at 10:00 or Doctor B at 11:00', context)
        except RouterOutputError as error:
            result = error
        assert context == original
        return result, requests, router.calls


def assert_scope_reminder(requests):
    first, second = [request['messages'] for request in requests]
    assert second[:-1] == first
    reminder = second[-1]
    assert reminder['role'] == 'system'
    assert 'field' in reminder['content'].lower()
    assert 'proposed_ops' in reminder['content']
    for rejected_data in ('Doctor A', 'Doctor B', '10:00', '11:00'):
        assert rejected_data not in reminder['content']
    assert not any(message['role'] == 'assistant' for message in second)


@pytest.mark.parametrize('kind', ['ambiguous_value', 'unsupported_value'])
def test_null_scope_remains_invalid_but_retry_may_supply_one_configured_field(config, tmp_path, kind):
    bad, good = payload(kind), payload(kind, 'doctor')
    with pytest.raises(ValueError):
        parse_response(json.dumps(bad), config)
    state = ScopedTaskState(config)
    before = take_snapshot(state)
    result, requests, calls = asyncio.run(run_router(config, tmp_path, state.router_context(), [bad, good]))
    assert result.clarification.slot == 'doctor'
    assert result.ops == [] and result.proposed_ops == []
    assert take_snapshot(state) == before
    assert len(calls) == 2 and not calls[0]['ok'] and calls[1]['ok']
    assert calls[0]['error'] == 'ValidationError' and calls[0]['failure_kind'] == 'model_output'
    assert calls[0]['response_shape']['clarification_slot'] is None
    assert_scope_reminder(requests)
    action = state.consume(result.model_dump())
    assert action.slot == 'doctor' and state.values == {} and state.pending_proposal is None


@pytest.mark.parametrize('kind', ['ambiguous_value', 'unsupported_value'])
def test_repeated_null_scope_exhausts_retry_without_applying_or_inventing_facts(config, tmp_path, kind):
    state = ScopedTaskState(config)
    state.apply([{'op': 'set', 'slot': 'date', 'value': '2026-09-15'}])
    before = take_snapshot(state)
    result, requests, calls = asyncio.run(run_router(config, tmp_path, state.router_context(), [payload(kind)]))
    assert isinstance(result, RouterOutputError)
    assert len(calls) == 2 and all(not call['ok'] for call in calls)
    assert take_snapshot(state) == before
    assert_scope_reminder(requests)


@pytest.mark.parametrize('kind', ['ambiguous_value', 'unsupported_value'])
def test_retry_new_scope_cannot_replace_an_existing_unresolved_question(config, tmp_path, kind):
    state = ScopedTaskState(config)
    state.apply([{'op': 'set', 'slot': 'date', 'value': '2026-09-15'}])
    state.consume(parse_response(json.dumps(payload('ambiguous_value', 'doctor')), config).model_dump())
    pending = deepcopy(state.pending_clarification)
    before = take_snapshot(state)
    result, requests, calls = asyncio.run(run_router(config, tmp_path, state.router_context(),
        [payload(kind), payload(kind, 'time')]))
    assert len(calls) == 2 and calls[-1]['ok']
    assert_scope_reminder(requests)
    action = state.consume(result.model_dump())
    assert state.pending_clarification == pending
    assert action.slot == 'doctor'
    assert take_snapshot(state) == before
    assert state.pending_proposal is None and not state.confirmed


def test_pizza_schema_failure_does_not_receive_flat_scope_guidance(monkeypatch, tmp_path):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    config = demo_config(load_config(ROOT / 'configs/pizza.json'))
    config['router']['retries'] = 1
    state = DemoOrderState(config)
    before = take_snapshot(state)
    result, requests, calls = asyncio.run(run_router(config, tmp_path, state.router_context(), [payload()]))
    assert isinstance(result, RouterOutputError) and len(calls) == 2
    assert requests[0]['messages'] == requests[1]['messages']
    assert take_snapshot(state) == before
    # Pizza's own legal unknown scope remains legal under its distinct schema.
    result, requests, calls = asyncio.run(run_router(config, tmp_path, state.router_context(),
        [payload('order_details')]))
    assert result.clarification.slot is None and len(calls) == 1 and calls[0]['ok']
