"""Exact configured alternatives are conservative output checks, not language accuracy."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.enum_grounding import EnumAlternativeError, unresolved_enum_alternatives, validate_enum_alternatives
from engine.router import Router, RouterOutputError, parse_response
from engine.scoped_task import ScopedTaskState


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    result = demo_config(load_config(ROOT / 'configs/clinic.json'))
    result['router']['retries'] = 1
    return result


def payload(**changes):
    result = dict(intent='task', ops=[], confidence=.99, unclear=False,
        is_affirmation=False, is_negation=False, clarification=None, proposed_ops=[],
        resolves_clarification=None, discard_clarification=None, discard_request=None)
    result.update(changes)
    return result


def choose(field='doctor', value='doctor_a'):
    return dict(op='set', slot=field, value=value)


def question(field='doctor', **changes):
    return payload(intent='ambiguous', clarification=dict(kind='ambiguous_value', slot=field, item_ids=[]), **changes)


async def routed(config, tmp_path, text, candidates, state, observed):
    before = deepcopy(state)
    def respond(request):
        observed.setdefault('requests', []).append(json.loads(request.content))
        value = candidates[min(len(observed['requests']) - 1, len(candidates) - 1)]
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps(value)}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        router = Router(config, tmp_path, api_key='offline-fixture', client=client)
        router.limiter.reserve = lambda: None
        try:
            return await router.route(text, state)
        finally:
            observed['calls'] = deepcopy(router.calls)
            assert state == before


@pytest.mark.parametrize('text', ['doctor A or doctor B', 'doctor_a ou doctor_b.',
    'الطبيب ألف ولا الطبيب باء؟', 'ألف أو باء', 'ألف او باء',
    'Doctor A or doctor B. Date 2026-09-15. Time 10:30.'])
def test_exact_configured_choices_with_explicit_independent_fields(config, text):
    assert unresolved_enum_alternatives(text, config) == {'doctor'}


@pytest.mark.parametrize('text', ['doctor A or doctor B, choose second',
    'doctor A or doctor B. choose second', 'doctor A or doctor B. Actually doctor A.',
    'not doctor A or doctor B', 'doctor A at 10 or doctor B at 11',
    'doctor A or doctor B. Time 10 with doctor B', 'doctor A or doctor B. tomorrow',
    'xdoctor A or doctor B', 'doctor A or doctor Bextra', 'doctor A or doctor_a'])
def test_selection_correction_dependent_pairs_boundaries_and_same_choice_abstain(config, text):
    assert unresolved_enum_alternatives(text, config) == set()


def test_alias_collision_does_not_establish_a_distinct_choice(config):
    slot = next(slot for slot in config['slots'] if slot['id'] == 'doctor')
    slot['aliases']['doctor_a'].append('shared')
    slot['aliases']['doctor_b'].append('shared')
    assert unresolved_enum_alternatives('shared or doctor A', config) == set()
    assert unresolved_enum_alternatives('doctor A or doctor B', config) == {'doctor'}


def test_custom_field_and_choices_are_not_clinic_constants(config, tmp_path):
    custom = json.loads(json.dumps(config).replace('doctor', 'counter'))
    observed = {}
    result = asyncio.run(routed(custom, tmp_path, 'counter A or counter B. Date 2026-09-15.',
        [payload(ops=[choose('counter', 'counter_a')]), question('counter',
            proposed_ops=[choose('date', '2026-09-15')])], {'state': {}}, observed))
    assert result.ops == [] and result.clarification.slot == 'counter'
    assert len(observed['requests']) == 2
    task = ScopedTaskState(custom)
    task.consume(result.model_dump())
    assert task.values == {} and task.pending_proposal['state'] == {'date': '2026-09-15'}


@pytest.mark.parametrize('candidate', [payload(ops=[choose()]),
    payload(ops=[choose('date', '2026-09-15')]), payload(is_affirmation=True),
    question('date', proposed_ops=[choose()])])
def test_commit_affirmation_and_staging_unresolved_choice_are_rejected(config, candidate):
    parsed = parse_response(json.dumps(candidate), config)
    with pytest.raises(EnumAlternativeError):
        validate_enum_alternatives(parsed, 'doctor A or doctor B', config)


def test_retry_is_static_bounded_and_only_valid_output_can_create_proposal(config, tmp_path):
    observed, task = {}, ScopedTaskState(config)
    bad = payload(ops=[choose(), choose('date', '2026-09-15'), choose('time', '10:30')])
    good = question(proposed_ops=[choose('date', '2026-09-15'), choose('time', '10:30')])
    text = 'doctor A or doctor B. Date 2026-09-15. Time 10:30.'
    result = asyncio.run(routed(config, tmp_path, text, [bad, good], task.router_context(), observed))
    assert task.values == {} and task.pending_proposal is None
    assert len(observed['calls']) == 2
    assert observed['calls'][0]['validation_rule'] == 'explicit_enum_alternatives'
    assert observed['calls'][0]['failure_kind'] == 'model_output'
    assert not observed['calls'][0]['ok'] and observed['calls'][1]['ok']
    first, second = [request['messages'] for request in observed['requests']]
    assert second[:-1] == first and second[-1]['role'] == 'system'
    assert not any(value in second[-1]['content'] for value in ['doctor_a', '2026-09-15', '10:30'])
    task.consume(result.model_dump())
    assert task.values == {} and task.pending_proposal['state'] == {'date': '2026-09-15', 'time': '10:30'}
    assert not task.confirmed


def test_exhausted_bad_outputs_do_not_touch_existing_pending_proposal(config, tmp_path):
    task = ScopedTaskState(config)
    task.consume(question(proposed_ops=[choose('date', '2026-09-15')]))
    before = deepcopy(task.__dict__)
    observed = {}
    with pytest.raises(RouterOutputError):
        asyncio.run(routed(config, tmp_path, 'doctor A or doctor B',
            [payload(ops=[choose()], resolves_clarification=task.pending_clarification['id'])],
            task.router_context(), observed))
    assert len(observed['calls']) == 2 and all(not call['ok'] for call in observed['calls'])
    assert task.__dict__ == before


def test_later_explicit_answer_can_resolve_existing_proposal(config, tmp_path):
    task = ScopedTaskState(config)
    task.consume(question(proposed_ops=[choose('date', '2026-09-15')]))
    observed = {}
    result = asyncio.run(routed(config, tmp_path, 'doctor A or doctor B. choose doctor B',
        [payload(ops=[choose(value='doctor_b')], resolves_clarification=task.pending_clarification['id'])],
        task.router_context(), observed))
    assert len(observed['calls']) == 1
    task.consume(result.model_dump())
    assert task.values == {'doctor': 'doctor_b', 'date': '2026-09-15'}
    assert task.pending_proposal is None


def test_base_and_pizza_modes_are_outside_guard(config):
    config['demo']['state_kind'] = 'pizza_order'
    assert unresolved_enum_alternatives('doctor A or doctor B', config) == set()
    config.pop('demo')
    assert unresolved_enum_alternatives('doctor A or doctor B', config) == set()
