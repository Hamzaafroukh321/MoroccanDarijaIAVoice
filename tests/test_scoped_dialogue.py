"""Canonical cross-domain regressions, not native speech or booking validation."""

from copy import deepcopy
import json
from jsonschema import Draft202012Validator

import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.demo_order import DemoOrderState
from engine.dialogue import ScopedDialogue
from engine.router import build_messages, parse_response, response_format
from engine.scoped_task import ScopedTaskState


def payload(**updates):
    result = dict(ops=[], confidence=.95, unclear=False, is_affirmation=False,
                  is_negation=False, intent='task', clarification=None,
                  resolves_clarification=None, proposed_ops=[], discard_clarification=None)
    result.update(updates)
    return result


@pytest.fixture(params=['pizza', 'clinic'])
def case(request):
    config = demo_config(load_config(ROOT / 'configs' / f'{request.param}.json'))
    if request.param == 'pizza':
        state = DemoOrderState(config)
        def op(kind, slot=None, value=None, item_id=None):
            return dict(op=kind, slot=slot, value=value, item_id=item_id)
        proposal = [op('create', item_id=1), op('set', 'quantity', 1, 1),
                    op('set', 'size', 'large', 1), op('set', 'toppings', ['cheese'], 1)]
        clarification = dict(kind='unsupported_drink', slot='drink', item_ids=[])
        resolve = op('set', 'drink', ['cola'])
        unrelated = op('set', 'size', 'small', 1)
        invalid = op('set', 'size', 'small', 99)
        baseline = op('set', 'drink', ['water'])
        expected = {'items': [{'id': 1, 'quantity': 1, 'size': 'large', 'toppings': ['cheese']}], 'drink': ['cola']}
        alternate = payload(intent='unclear', unclear=True,
                            clarification=dict(kind='unintelligible', slot='size', item_ids=[]))
    else:
        state = ScopedTaskState(config)
        def op(kind, slot=None, value=None):
            return dict(op=kind, slot=slot, value=value)
        proposal = [op('set', 'date', '15/09/2026'), op('set', 'time', '9h30')]
        clarification = dict(kind='unsupported_value', slot='doctor', item_ids=[])
        resolve = op('set', 'doctor', 'doctor_a')
        unrelated = op('set', 'date', '2026-09-16')
        invalid = op('set', 'time', '25:99')
        baseline = op('set', 'doctor', 'doctor_b')
        expected = {'doctor': 'doctor_a', 'date': '2026-09-15', 'time': '09:30'}
        alternate = payload(intent='ambiguous',
                            clarification=dict(kind='ambiguous_value', slot='date', item_ids=[]))
    return dict(domain=request.param, config=config, state=state, proposal=proposal,
                clarification=clarification, resolve=resolve, unrelated=unrelated,
                invalid=invalid, baseline=baseline, expected=expected, alternate=alternate)


def stage(case):
    response = payload(intent='out_of_scope', clarification=case['clarification'], proposed_ops=case['proposal'])
    parsed = parse_response(json.dumps(response), case['config']).model_dump()
    return case['state'].consume(parsed)


def test_paired_staging_keeps_facts_and_uses_shared_lifecycle_then_fresh_confirmation(case):
    state = case['state']
    assert type(state).consume is ScopedDialogue.consume
    before, version = deepcopy(state.values), state.version
    assert stage(case).kind == case['clarification']['kind']
    assert state.values == before and state.version == version
    pending_id = state.pending_clarification['id']
    proposal = deepcopy(state.pending_proposal)
    assert proposal['base_version'] == version
    state.begin_confirmation(version)
    assert state.readback_version is None and not state.confirmed
    resolution = payload(ops=[case['resolve']], resolves_clarification=pending_id, is_affirmation=True)
    action = state.consume(parse_response(json.dumps(resolution), case['config']).model_dump())
    assert action.kind == 'readback'
    assert state.values == case['expected']
    assert state.pending_clarification is None and state.pending_proposal is None
    assert not state.confirmed and state.readback_version is None
    state.begin_confirmation(version)
    assert state.readback_version is None
    assert state.consume(payload(is_affirmation=True)).kind == 'readback'
    state.begin_confirmation(state.version)
    assert state.consume(payload(is_affirmation=True)).kind == 'accepted'
    assert state.confirmed


def test_wrong_scope_and_invalid_combined_batch_preserve_proposal_and_all_state(case):
    state = case['state']
    stage(case)
    pending, proposal = deepcopy(state.pending_clarification), deepcopy(state.pending_proposal)
    before, version, repairs = deepcopy(state.values), state.version, state.repair_count
    for operations in ([case['unrelated']], [case['resolve'], case['invalid']]):
        with pytest.raises(ValueError):
            state.consume(payload(ops=operations, resolves_clarification=pending['id']))
        assert state.values == before and state.version == version and state.repair_count == repairs
        assert state.pending_proposal == proposal and state.pending_clarification == pending
        assert not state.confirmed
    with pytest.raises(ValueError, match='staged proposal'):
        state.consume(payload(ops=[case['unrelated']]))
    assert state.values == before and state.pending_proposal == proposal
    if case['domain'] == 'pizza':
        assert state.next_item_id == 1


def test_stale_resolution_ids_reject_without_reapplying_old_facts(case):
    state = case['state']
    stage(case)
    first_id = state.pending_clarification['id']
    state.consume(payload(ops=[case['resolve']], resolves_clarification=first_id))
    state.consume(payload(intent='out_of_scope', clarification=case['clarification']))
    second = deepcopy(state.pending_clarification)
    before, version = deepcopy(state.values), state.version
    assert second['id'] > first_id
    with pytest.raises(ValueError, match='stale'):
        state.consume(payload(ops=[case['resolve']], resolves_clarification=first_id))
    assert state.values == before and state.version == version and state.pending_clarification == second


def test_explicit_discard_preserves_previously_committed_data_and_ids(case):
    state = case['state']
    state.apply([case['baseline']])
    before, version = deepcopy(state.values), state.version
    stage(case)
    pending_id = state.pending_clarification['id']
    response = payload(discard_clarification=pending_id)
    state.consume(parse_response(json.dumps(response), case['config']).model_dump())
    assert state.values == before and state.version > version
    assert state.pending_clarification is None and state.pending_proposal is None
    assert state.readback_version is None and not state.confirmed
    if case['domain'] == 'pizza':
        assert state.next_item_id == 1
    with pytest.raises(ValueError, match='stale'):
        state.consume(response)
    assert state.values == before


def test_initial_pending_question_and_proposal_cannot_be_replaced(case):
    state = case['state']
    stage(case)
    pending, proposal = deepcopy(state.pending_clarification), deepcopy(state.pending_proposal)
    assert state.consume(case['alternate']).kind == pending['kind']
    assert state.pending_clarification == pending and state.pending_proposal == proposal
    assert state.repair_count == 2
    with pytest.raises(ValueError, match='initial'):
        stage(case)
    assert state.pending_clarification == pending and state.pending_proposal == proposal


def test_failed_confirmation_during_pending_issue_is_bounded(case):
    state = case['state']
    before = deepcopy(state.values)
    stage(case)
    for _ in range(case['config']['demo']['max_repairs'] - 1):
        assert state.consume(payload(is_affirmation=True)).kind == case['clarification']['kind']
    assert state.consume(payload(is_negation=True)).kind == 'handoff'
    assert state.values == before and state.pending_proposal is not None
    assert not state.confirmed


@pytest.fixture
def clinic_config():
    return demo_config(load_config(ROOT / 'configs/clinic.json'))


def test_clinic_production_schema_is_strict_and_has_only_configured_fields(clinic_config):
    format = response_format(clinic_config)
    assert format['type'] == 'json_schema' and format['json_schema']['strict'] is True
    schema = format['json_schema']['schema']
    for value in [schema, *schema['$defs'].values()]:
        if value.get('type') == 'object':
            assert value['additionalProperties'] is False
            assert set(value['required']) == set(value['properties'])
    assert set(schema['$defs']['Operation']['properties']['slot']['enum']) == {'doctor', 'date', 'time'}
    assert {slot for slot in ('doctor', 'date', 'time', 'unknown', None)
            if Draft202012Validator(schema['properties']['clarification']).is_valid(
                {'kind': 'ambiguous_value', 'slot': slot, 'item_ids': []})} == {'doctor', 'date', 'time'}
    valid = payload(ops=[dict(op='set', slot='doctor', value='doctor_a')])
    assert parse_response(json.dumps(valid), clinic_config).ops[0].value == 'doctor_a'
    for field in valid:
        incomplete = dict(valid)
        incomplete.pop(field)
        with pytest.raises(ValueError):
            parse_response(json.dumps(incomplete), clinic_config)
    with pytest.raises(ValueError):
        parse_response(json.dumps({**valid, 'booking_reference': 'fabricated'}), clinic_config)


@pytest.mark.parametrize('operation', [dict(op='set', slot='doctor', value='unlisted_doctor'),
    dict(op='set', slot='size', value='large'), dict(op='set', slot='time', value='25:99'),
    dict(op='set', slot='date', value='not a date')])
def test_clinic_production_parser_rejects_invalid_facts(clinic_config, operation):
    with pytest.raises(ValueError):
        parse_response(json.dumps(payload(ops=[operation])), clinic_config)


def test_clinic_production_messages_use_pending_preview_and_bounded_real_history(clinic_config):
    state = ScopedTaskState(clinic_config)
    state.consume(payload(intent='out_of_scope',
        clarification=dict(kind='unsupported_value', slot='doctor', item_ids=[]),
        proposed_ops=[dict(op='set', slot='date', value='2026-09-15'),
                      dict(op='set', slot='time', value='09:30')]))
    context = state.router_context()
    context['pending_request'] = {'text': 'unsupported doctor with specified date and time ' * 40, 'truncated': False}
    context['last_assistant_text'] = 'Please choose fictional doctor A or B. ' * 20
    before = deepcopy(context)
    messages = build_messages(clinic_config, context, 'doctor_a')
    assert context == before and state.values == {}
    assert [message['role'] for message in messages] == ['system', 'user', 'assistant', 'user']
    assert messages[1]['content'] == context['pending_request']['text'][:1000]
    assert messages[2]['content'] == context['last_assistant_text'][:400]
    assert messages[-1]['content'] == 'doctor_a'
    prompt = messages[0]['content']
    assert 'pizza' not in prompt.casefold() and 'toppings' not in prompt.casefold()
    assert 'fictional' in prompt and 'no availability check or booking' in prompt
    effective = json.loads(prompt.split('CONTEXT: ', 1)[1].split('\nUTTERANCE:', 1)[0])
    assert effective['state'] == {'date': '2026-09-15', 'time': '09:30'}
    assert effective['committed_state'] == {}
    assert effective['requested_slot'] == 'doctor'
    assert effective['pending_request']['truncated'] is True
    assert effective['pending_clarification']['id'] == state.pending_clarification['id']


def test_clinic_cannot_stage_an_answer_to_its_own_unresolved_field(clinic_config):
    state = ScopedTaskState(clinic_config)
    invalid = payload(intent='out_of_scope',
                      clarification=dict(kind='unsupported_value', slot='doctor', item_ids=[]),
                      proposed_ops=[dict(op='set', slot='doctor', value='doctor_a')])
    with pytest.raises(ValueError):
        parse_response(json.dumps(invalid), clinic_config)
    with pytest.raises(ValueError):
        state.consume(invalid)
    assert state.values == {} and state.pending_proposal is None and state.pending_clarification is None
