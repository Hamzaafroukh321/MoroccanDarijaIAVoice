"""Pending answers require their explicit ID; saved live-payload regression."""

import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.demo_order import DemoOrderState
from engine.router import Router
from engine.scoped_task import ScopedTaskState


def payload(**changes):
    result = dict(ops=[], confidence=1.0, unclear=False, is_affirmation=False,
                  is_negation=False, intent='task', clarification=None,
                  resolves_clarification=None, proposed_ops=[], discard_clarification=None)
    result.update(changes)
    return result


def case(domain):
    config = demo_config(load_config(ROOT / 'configs' / f'{domain}.json'))
    if domain == 'pizza':
        state = DemoOrderState(config)
        clarification = dict(kind='drink_size', slot='drink', item_ids=[])
        # Exact operation ordering/values from saved live turn29ac1fc5...:
        details = [dict(op='create', item_id=1, slot=None, value=None),
                   dict(op='set', item_id=1, slot='size', value='large'),
                   dict(op='set', item_id=1, slot='toppings', value=['cheese']),
                   dict(op='set', item_id=1, slot='quantity', value=2)]
        answer = dict(op='set', item_id=None, slot='drink', value=['water'])
    else:
        state = ScopedTaskState(config)
        clarification = dict(kind='unsupported_value', slot='doctor', item_ids=[])
        details = [dict(op='set', slot='date', value='2026-09-15'),
                   dict(op='set', slot='time', value='10:30')]
        answer = dict(op='set', slot='doctor', value='doctor_a')
    return config, state, clarification, details, answer


def snapshot(state):
    return deepcopy((state.values, state.version, getattr(state, 'next_item_id', None),
                     state.pending_clarification, state.pending_proposal,
                     state.readback_version, state.confirmed_version))


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
@pytest.mark.parametrize('staged', [False, True])
@pytest.mark.parametrize('resolution', [None, 99])
def test_answer_without_correct_pending_id_rejects_entire_batch(domain, staged, resolution):
    _, state, clarification, details, answer = case(domain)
    state.consume(payload(intent='out_of_scope', clarification=clarification,
                          proposed_ops=details if staged else []))
    original = snapshot(state)
    operations = [answer] if staged else details + [answer]
    with pytest.raises(ValueError, match='(?i)clarification|resolv|pending|proposal'):
        state.consume(payload(ops=operations, resolves_clarification=resolution))
    assert snapshot(state) == original and not state.confirmed
    # The exact corrected ID, explicitly supplied by the router, commits once
    # and still requires a newly completed readback before any confirmation.
    result = state.consume(payload(ops=operations,
                                  resolves_clarification=state.pending_clarification['id']))
    assert result.kind == 'readback' and state.ready and not state.confirmed
    assert state.pending_clarification is None and state.pending_proposal is None
    assert state.readback_version is None
    state.begin_confirmation(state.version)
    assert state.consume(payload(is_affirmation=True)).kind == 'accepted'
    if domain == 'pizza':
        assert state.next_item_id == 2
        assert state.values == {'items': [{'id': 1, 'size': 'large', 'toppings': ['cheese'], 'quantity': 2}],
                                'drink': ['water']}


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_unrelated_edit_without_proposal_remains_allowed_and_keeps_pending_question(domain):
    _, state, clarification, details, _ = case(domain)
    state.consume(payload(intent='out_of_scope', clarification=clarification))
    pending = deepcopy(state.pending_clarification)
    result = state.consume(payload(ops=details))
    assert result.kind == clarification['kind'] and state.pending_clarification == pending
    assert state.values and state.version == 1 and not state.confirmed
    if domain == 'pizza':
        assert 'drink' not in state.values
    else:
        assert 'doctor' not in state.values


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
@pytest.mark.parametrize('staged', [False, True])
@pytest.mark.parametrize('initial_id', [None, 99])
def test_router_retries_answer_identity_mismatch_before_returning_any_operations(tmp_path, domain, staged, initial_id):
    config, state, clarification, details, answer = case(domain)
    config['router']['retries'] = 1
    state.consume(payload(intent='out_of_scope', clarification=clarification,
                          proposed_ops=details if staged else []))
    original = snapshot(state)
    operations = [answer] if staged else details + [answer]
    attempts = []
    def endpoint(request):
        attempts.append(json.loads(request.content))
        result = payload(ops=operations, resolves_clarification=(initial_id if len(attempts) == 1 else
                          state.pending_clarification['id']))
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps(result)}}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
            router = Router(config, tmp_path, api_key='offline-fixture', client=client)
            router.limiter.reserve = lambda: None
            result = await router.route('explicit supported answer', state.router_context())
            assert len(attempts) == 2
            assert result.resolves_clarification == state.pending_clarification['id']
            assert snapshot(state) == original
            assert router.calls[0]['failure_kind'] == 'model_output'
            assert router.calls[1]['ok'] is True
            state.consume(result.model_dump())
            assert state.ready and state.pending_clarification is None and not state.confirmed
    asyncio.run(run())
