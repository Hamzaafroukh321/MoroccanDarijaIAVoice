"""Narrow numeric time grounding is output validation, not ASR accuracy."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.normalize import normalize
from engine.router import Router, RouterOutputError


def response(value='08:00', **changes):
    payload = dict(intent='task', ops=[{'op': 'set', 'slot': 'time', 'value': value}],
        confidence=.99, unclear=False, is_affirmation=False, is_negation=False,
        clarification=None, proposed_ops=[], resolves_clarification=None,
        discard_clarification=None, discard_request=None)
    payload.update(changes)
    return payload


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    result = demo_config(load_config(ROOT / 'configs/clinic.json'))
    result['router']['retries'] = 1
    return result


async def route(config, tmp_path, text, payloads, state=None):
    requests = []
    state = state if state is not None else {'state': {'doctor': 'doctor_a',
        'date': '2026-09-15', 'time': '10:30'}, 'requested_slot': 'time'}
    original = deepcopy(state)
    def handle(request):
        requests.append(json.loads(request.content))
        payload = payloads[min(len(requests) - 1, len(payloads) - 1)]
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps(payload)}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        router = Router(config, tmp_path, api_key='offline-fixture', client=client)
        router.limiter.reserve = lambda: None
        try:
            parsed = await router.route(text, state)
        finally:
            assert state == original
        return parsed, router.calls, requests


def test_real_saved_darija_correction_cannot_change_eleven_to_eight(config, tmp_path):
    transcript = normalize('لا بدل ساعة خليه مع الحداش', config)
    assert '11' in transcript
    parsed, calls, requests = asyncio.run(route(config, tmp_path, transcript,
        [response('08:00'), response('11:00')]))
    assert parsed.ops[0].value == '11:00'
    assert len(requests) == 2
    assert calls[0]['failure_kind'] == 'model_output' and not calls[0]['ok']
    assert calls[1]['ok']


def test_exact_clock_minutes_must_match_before_output_is_accepted(config, tmp_path):
    parsed, calls, _ = asyncio.run(route(config, tmp_path, 'بدل ساعة خليه مع 11:30',
        [response('11:00'), response('11:30')]))
    assert parsed.ops[0].value == '11:30'
    assert len(calls) == 2 and not calls[0]['ok']


def test_exhausted_wrong_hour_outputs_use_existing_bounded_recovery(config, tmp_path):
    with pytest.raises(RouterOutputError):
        asyncio.run(route(config, tmp_path, 'بدل ساعة خليه مع 11', [response('08:00')]))


def test_uncertain_time_may_return_clarification_instead_of_guess(config, tmp_path):
    clarify = response(ops=[], intent='ambiguous', clarification={
        'kind': 'ambiguous_value', 'slot': 'time', 'item_ids': []})
    parsed, calls, _ = asyncio.run(route(config, tmp_path, 'بدل ساعة خليه مع 11',
        [response('08:00'), clarify]))
    assert parsed.ops == [] and parsed.clarification.slot == 'time'
    assert len(calls) == 2


@pytest.mark.parametrize('text,requested_slot', [('11', 'time'), ('at 11', 'doctor'),
    ('no change time 11', 'time'), ('لا بدل ساعة خليه مع 11', 'doctor')])
def test_supported_hour_cues_reject_wrong_hour(config, tmp_path, text, requested_slot):
    parsed, calls, _ = asyncio.run(route(config, tmp_path, text,
        [response('08:00'), response('23:30')], state={'state': {}, 'requested_slot': requested_slot}))
    assert parsed.ops[0].value == '23:30'
    assert len(calls) == 2


@pytest.mark.parametrize('text', ['الساعة 11', 'الساعة 11.'])
@pytest.mark.parametrize('value', ['11:00', '11:30', '23:45'])
def test_hour_only_does_not_invent_minutes_or_am_pm(config, tmp_path, text, value):
    parsed, calls, _ = asyncio.run(route(config, tmp_path, text, [response(value)]))
    assert parsed.ops[0].value == value and len(calls) == 1


@pytest.mark.parametrize('text', ['ماشي مع 11', 'لا مع 11', 'not at 11', 'before 11',
    'between 10 and 11', 'at 10 or 11', 'موعد 2026-09-11', 'نهار 11/09',
    'quantity 11', 'رقم الهاتف 0611223344', 'بدل ساعة وخذ كمية 11', 'مع 11 العشية'])
def test_ambiguous_excluded_or_non_time_numbers_are_not_grounded(config, tmp_path, text):
    parsed, calls, _ = asyncio.run(route(config, tmp_path, text, [response('08:00')]))
    assert parsed.ops[0].value == '08:00' and len(calls) == 1


def test_bare_numeric_reply_can_use_pending_time_scope(config, tmp_path):
    state = {'state': {}, 'requested_slot': 'doctor', 'pending_clarification': {
        'id': 7, 'kind': 'ambiguous_value', 'slot': 'time', 'item_ids': []}}
    parsed, calls, _ = asyncio.run(route(config, tmp_path, '11',
        [response('08:00', resolves_clarification=7), response('11:00', resolves_clarification=7)], state))
    assert parsed.ops[0].value == '11:00' and len(calls) == 2


def test_guard_uses_configured_time_type_not_literal_field_name(config, tmp_path):
    next(slot for slot in config['slots'] if slot['id'] == 'time')['id'] = 'arrival'
    wrong, right = response('08:00'), response('11:00')
    wrong['ops'][0]['slot'] = right['ops'][0]['slot'] = 'arrival'
    parsed, calls, _ = asyncio.run(route(config, tmp_path, '11', [wrong, right],
        state={'state': {}, 'requested_slot': 'arrival'}))
    assert parsed.ops[0].slot == 'arrival' and len(calls) == 2


def test_current_transcript_only_does_not_ground_from_retained_request(config, tmp_path):
    state = {'state': {}, 'requested_slot': 'time', 'pending_request': {
        'id': 'request_1', 'text': 'الساعة 11', 'clarification_resolved': True},
        'last_assistant_text': 'Which time?'}
    _, calls, _ = asyncio.run(route(config, tmp_path, 'eight please', [response('08:00')], state))
    assert len(calls) == 1


def test_configured_time_format_is_canonicalized_and_retry_feedback_is_static(config, tmp_path):
    assert '%Hh%M' in config['normalization']['time_input_formats']
    parsed, calls, requests = asyncio.run(route(config, tmp_path, 'at 11:30',
        [response('08h00'), response('11h30')]))
    assert parsed.ops[0].value == '11h30'
    assert len(calls) == 2
    assert calls[0]['validation_rule'] == 'current_utterance_time'
    first, second = [request['messages'] for request in requests]
    assert second[:-1] == first
    feedback = second[-1]
    assert feedback['role'] == 'system'
    assert 'explicit clock value' in feedback['content']
    assert '08h00' not in feedback['content'] and '11h30' not in feedback['content']
    assert 'ops' not in feedback['content']
    assert not any(message['role'] == 'assistant' for message in second)


@pytest.mark.parametrize('text', ['at 11 pm', 'at 11am', 'at 11 a.m.', 'at 11:xx'])
def test_am_pm_suffix_or_malformed_clock_is_outside_numeric_guard(config, tmp_path, text):
    parsed, calls, _ = asyncio.run(route(config, tmp_path, text, [response('08:00')]))
    assert parsed.ops[0].value == '08:00' and len(calls) == 1


def test_multiple_configured_time_fields_do_not_guess_which_time_is_grounded(config, tmp_path):
    second_time = deepcopy(next(slot for slot in config['slots'] if slot['type'] == 'time'))
    second_time['id'] = 'departure'
    config['slots'].append(second_time)
    parsed, calls, _ = asyncio.run(route(config, tmp_path, 'at 11:30', [response('08:00')]))
    assert parsed.ops[0].value == '08:00' and len(calls) == 1


@pytest.mark.parametrize('text', ["don't schedule at 11", 'don’t schedule at 11',
    'never at 11', "won't schedule at 11", 'cannot schedule at 11', 'avoid at 11',
    'exclude at 11', 'cancel at 11', 'مابغيتش مع 11'])
def test_excluded_time_does_not_become_positive_grounding(config, tmp_path, text):
    parsed, calls, requests = asyncio.run(route(config, tmp_path, text, [response('08:00')]))
    assert parsed.ops[0].value == '08:00'
    assert len(calls) == len(requests) == 1
    assert 'validation_rule' not in calls[0]


def test_unmarked_clock_allows_pm_but_requires_stated_minutes(config, tmp_path):
    parsed, calls, _ = asyncio.run(route(config, tmp_path, 'at 11:30',
        [response('23:00'), response('23:30')]))
    assert parsed.ops[0].value == '23:30'
    assert len(calls) == 2
    assert calls[0]['validation_rule'] == 'current_utterance_time'


@pytest.mark.parametrize('source,wrong,right', [('23:30', '11:30', '23:30'),
                                              ('00:30', '12:30', '00:30')])
def test_explicit_twenty_four_hour_values_are_not_reduced_modulo_twelve(config, tmp_path, source, wrong, right):
    parsed, calls, _ = asyncio.run(route(config, tmp_path, 'at ' + source,
        [response(wrong), response(right)]))
    assert parsed.ops[0].value == right and len(calls) == 2


@pytest.mark.parametrize('retry_kind', ['correct_proposal', 'clarify_time'])
def test_staged_time_must_be_grounded_before_it_can_be_saved_as_a_proposal(config, tmp_path, retry_kind):
    wrong = response(intent='out_of_scope', ops=[], proposed_ops=[
        {'op': 'set', 'slot': 'time', 'value': '08:00'}], clarification={
        'kind': 'unsupported_value', 'slot': 'doctor', 'item_ids': []})
    corrected = deepcopy(wrong)
    corrected['proposed_ops'][0]['value'] = '11:00'
    clarification = response(intent='ambiguous', ops=[], clarification={
        'kind': 'ambiguous_value', 'slot': 'time', 'item_ids': []})
    parsed, calls, _ = asyncio.run(route(config, tmp_path, 'other doctor at 11',
        [wrong, corrected if retry_kind == 'correct_proposal' else clarification]))
    assert len(calls) == 2 and calls[0]['validation_rule'] == 'current_utterance_time'
    assert parsed.ops == []
    if retry_kind == 'correct_proposal':
        assert parsed.proposed_ops[0].value == '11:00'
        assert parsed.clarification.slot == 'doctor'
    else:
        assert parsed.proposed_ops == [] and parsed.clarification.slot == 'time'


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_shared_draft_number_normalization_keeps_word_boundaries_and_review_gate(monkeypatch, domain):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    config = demo_config(load_config(ROOT / f'configs/{domain}.json'))
    assert normalize('ستة سبعة تمنية تسعود عشرة حضاش طناش تلطاش ربعطاش خمسطاش سطاش سبعطاش تمنطاش تسعطاش عشرين',
                     config) == '6 7 8 9 10 11 12 13 14 15 16 17 18 19 20'
    assert normalize('الوحدة الجوج التلاتة الربعة الخمسة الستة السبعة التمنية التسعود العشرة الحداش الطناش',
                     config) == '1 2 3 4 5 6 7 8 9 10 11 12'
    assert normalize('لا بدل ساعة خليه مع الحداش', config) == 'لا بدل ساعة خليه مع 11'
    assert normalize('بالحداش ربع', config) == 'بالحداش ربع'
    assert config['demo']['needs_human_review'] is True
