"""Dependent alternatives remain one uncommitted request until explicitly resolved."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config, reply_text
from engine.endpointing import Segment
from engine.pipeline import VoiceSession
from engine.recovery import RecoveryStore
from engine.responder import RenderedAudio
from engine.router import Router, parse_response
from engine.scoped_task import ScopedTaskState
from engine.stt import Transcript, wav_bytes


def response(**changes):
    value = dict(intent='task', ops=[], confidence=.99, unclear=False,
        is_affirmation=False, is_negation=False, clarification=None, proposed_ops=[],
        resolves_clarification=None, discard_clarification=None, discard_request=None)
    value.update(changes)
    return value


def op(slot, value):
    return dict(op='set', slot=slot, value=value)


def opening(field, **changes):
    return response(intent='ambiguous', clarification=dict(kind='ambiguous_value',
        slot=field, item_ids=[], coupled_slots=['time']), **changes)


@pytest.fixture(params=['clinic', 'custom'])
def case(request, monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    config = demo_config(load_config(ROOT / 'configs/clinic.json'))
    field = 'doctor'
    if request.param == 'custom':
        config = json.loads(json.dumps(config).replace('doctor', 'counter'))
        config['domain_id'] = 'service-desk'
        field = 'counter'
    task = ScopedTaskState(config)
    task.apply([op(field, field + '_a'), op('date', '2026-09-15'), op('time', '10:00')])
    return config, field, task


def test_partial_choice_preserves_committed_values_until_every_coupled_field_is_answered(case):
    config, field, task = case
    original, version = deepcopy(task.values), task.version
    task.begin_confirmation(version)
    task.consume(opening(field))
    first = task.pending_clarification['id']
    assert task.pending_proposal['ops'] == []
    assert task.pending_proposal['coupled_slots'] == [field, 'time']
    assert task.pending_proposal['answered_slots'] == []
    assert task.pending_proposal['remaining_slots'] == [field, 'time']
    assert task.readback_version is None and not task.confirmed
    action = task.consume(response(ops=[op(field, field + '_b')], resolves_clarification=first))
    assert task.values == original and task.version == version
    assert action.slot == 'time' and action.kind != 'readback'
    assert task.pending_clarification['id'] > first
    assert task.pending_proposal['answered_slots'] == [field]
    assert task.pending_proposal['remaining_slots'] == ['time']
    assert task.pending_proposal['state'] == {**original, field: field + '_b'}
    task.begin_confirmation(version)
    assert task.readback_version is None
    second = task.pending_clarification['id']
    action = task.consume(response(ops=[op('time', '11:00')], resolves_clarification=second))
    assert action.kind == 'readback'
    assert task.values == {**original, field: field + '_b', 'time': '11:00'}
    assert task.version == version + 1
    assert task.pending_proposal is None and task.pending_clarification is None
    assert task.readback_version is None and not task.confirmed
    with pytest.raises(ValueError):
        task.consume(response(ops=[op('time', '11:00')], resolves_clarification=second))
    assert task.version == version + 1
    assert task.consume(response(is_affirmation=True)).kind == 'readback'
    task.begin_confirmation(task.version)
    assert task.consume(response(is_affirmation=True)).kind == 'accepted'


def test_direct_complete_answer_and_independent_proposal_commit_once(case):
    _, field, task = case
    original, version = deepcopy(task.values), task.version
    task.consume(opening(field, proposed_ops=[op('date', '2026-09-16')]))
    assert task.values == original
    action = task.consume(response(ops=[op(field, field + '_b'), op('time', '11:00')],
        resolves_clarification=task.pending_clarification['id']))
    assert action.kind == 'readback'
    assert task.values == {field: field + '_b', 'date': '2026-09-16', 'time': '11:00'}
    assert task.version == version + 1 and task.pending_proposal is None


@pytest.mark.parametrize('failure', ['stale', 'missing_id', 'invalid_batch', 'unrelated'])
def test_invalid_partial_answer_preserves_entire_transaction(case, failure):
    _, field, task = case
    task.consume(opening(field))
    first = task.pending_clarification['id']
    task.consume(response(ops=[op(field, field + '_b')], resolves_clarification=first))
    current = task.pending_clarification['id']
    candidate = response(ops=[op('time', '11:00')], resolves_clarification=current)
    if failure == 'stale':
        candidate['resolves_clarification'] = first
    elif failure == 'missing_id':
        candidate['resolves_clarification'] = None
    elif failure == 'invalid_batch':
        candidate['ops'].append(op('date', 'not-a-date'))
    else:
        candidate = response(ops=[op('date', '2026-09-16')])
    snapshot = deepcopy(task.__dict__)
    with pytest.raises(ValueError):
        task.consume(candidate)
    assert task.__dict__ == snapshot


def test_bare_yes_no_and_repeated_failed_repairs_do_not_advance_coupled_choice(case):
    config, field, task = case
    task.consume(opening(field))
    first = task.pending_clarification['id']
    task.consume(response(ops=[op(field, field + '_b')], resolves_clarification=first))
    before = deepcopy((task.values, task.pending_proposal, task.pending_clarification))
    for index in range(config['demo']['max_repairs'] + 1):
        action = task.consume(response(**{'is_affirmation' if index % 2 else 'is_negation': True}))
        assert (task.values, task.pending_proposal, task.pending_clarification) == before
        assert not task.confirmed and task.readback_version is None
    assert action.kind == 'handoff'


def test_current_discard_drops_whole_draft_and_recovery_never_carries_staged_answer(case):
    config, field, task = case
    original = deepcopy(task.values)
    task.consume(opening(field))
    first = task.pending_clarification['id']
    task.consume(response(ops=[op(field, field + '_b')], resolves_clarification=first))
    registry = RecoveryStore(clock=lambda: 100)
    offer = registry.offer(config, task, 'offline-source', pending_request={'id': 'request_1', 'text': 'original'})
    assert offer['slots'] == original and offer['discarded_pending']
    restored = registry.restore(offer['token'], config).task
    assert restored.values == original and restored.pending_proposal is None
    assert restored.pending_clarification is None and not restored.confirmed
    before = deepcopy(task.__dict__)
    with pytest.raises(ValueError):
        task.consume(response(discard_clarification=first))
    assert task.__dict__ == before
    action = task.consume(response(discard_clarification=task.pending_clarification['id']))
    assert task.values == original and action.kind == 'readback'
    assert task.pending_proposal is None and task.pending_clarification is None
    assert task.readback_version is None


def test_initial_proposal_cannot_decide_any_coupled_field(case):
    _, field, task = case
    before = deepcopy(task.__dict__)
    with pytest.raises(ValueError):
        task.consume(opening(field, proposed_ops=[op('time', '11:00')]))
    assert task.__dict__ == before


def test_redundant_primary_answer_still_requires_new_time_answer(case):
    _, field, task = case
    task.consume(opening(field))
    task.consume(response(ops=[op(field, field + '_a')], resolves_clarification=task.pending_clarification['id']))
    assert task.pending_proposal['remaining_slots'] == ['time']
    assert task.next_action().slot == 'time' and not task.confirmed


@pytest.mark.parametrize('invalid', ['primary', 'duplicate', 'unknown', 'wrong_kind'])
def test_malformed_coupling_is_rejected_by_production_parser_before_state_change(case, invalid):
    config, field, task = case
    candidate = opening(field)
    if invalid == 'primary':
        candidate['clarification']['coupled_slots'] = [field]
    elif invalid == 'duplicate':
        candidate['clarification']['coupled_slots'] = ['time', 'time']
    elif invalid == 'unknown':
        candidate['clarification']['coupled_slots'] = ['unknown_field']
    else:
        candidate['intent'] = 'out_of_scope'
        candidate['clarification']['kind'] = 'unsupported_value'
    before = deepcopy(task.__dict__)
    with pytest.raises(ValueError):
        parse_response(json.dumps(candidate), config)
    assert task.__dict__ == before


def test_optional_but_explicitly_coupled_field_still_needs_new_answer(case):
    config, field, _ = case
    next(slot for slot in config['slots'] if slot['id'] == 'time')['required'] = False
    task = ScopedTaskState(config)
    task.apply([op(field, field + '_a'), op('date', '2026-09-15')])
    assert task.ready
    task.consume(opening(field))
    action = task.consume(response(ops=[op(field, field + '_b')],
        resolves_clarification=task.pending_clarification['id']))
    assert action.slot == 'time' and action.kind != 'readback'
    assert task.values[field] == field + '_a' and 'time' not in task.values
    assert task.pending_proposal['remaining_slots'] == ['time']


def test_pipeline_retains_original_until_final_commit_and_exact_new_playback_ack(case, tmp_path):
    async def run():
        config, field, initial = case
        current_text, candidates, requests, events = '', [], [], []
        ended = asyncio.Queue()
        def http(request):
            requests.append(json.loads(request.content))
            assert candidates, 'Unexpected extra router attempt'
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                'message': {'content': json.dumps(candidates.pop(0))}}]})
        class ASR:
            async def transcribe(self, pcm):
                return Transcript(current_text, .99, 'offline-fixture')
        class Bank:
            async def render(self, action, values):
                return RenderedAudio(wav_bytes(bytes(1024), config['runtime']), 512, 512,
                    ['fixture'], reply_text(action, values, config['demo']))
        async def emit(event):
            events.append(deepcopy(event))
            if event['type'] == 'audio_end':
                ended.put_nowait(event['playback_id'])
        async def audio(data):
            pass
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            router = Router(config, tmp_path, api_key='offline-fixture', client=client)
            router.limiter.reserve = lambda: None
            voice = VoiceSession(config, tmp_path, ASR(), router, Bank(), None, emit, audio)
            voice.task = initial
            async def acknowledge():
                playback_id = await asyncio.wait_for(ended.get(), 2)
                await voice.playback_finished(playback_id)
                await asyncio.wait_for(voice.playback_task, 2)
                return playback_id
            async def speak(text, candidate):
                nonlocal current_text
                current_text = text
                candidates.append(candidate)
                await voice.queue.put(Segment(bytes([len(requests) + 1, 0]) * 512, 0, 320, 1728, 320))
                await asyncio.wait_for(voice.queue.join(), 2)
            original = deepcopy(initial.values)
            text = f'{field} A at 10 or {field} B at 11'
            try:
                await voice.start()
                greeting = await acknowledge()
                await speak(text, opening(field))
                await acknowledge()
                retained = deepcopy(voice.pending_request)
                first = voice.task.pending_clarification['id']
                await speak(f'{field} B', response(ops=[op(field, field + '_b')], resolves_clarification=first))
                await acknowledge()
                assert voice.task.values == original
                assert voice.pending_request == retained
                assert voice.task.pending_clarification['id'] > first
                assert voice.task.pending_proposal['remaining_slots'] == ['time']
                second = voice.task.pending_clarification['id']
                await speak('11', response(ops=[op('time', '11:00')], resolves_clarification=second))
                fresh = await asyncio.wait_for(ended.get(), 2)
                assert text in json.dumps(requests[-1]['messages'])
                assert voice.pending_request is None
                assert voice.task.values == {**original, field: field + '_b', 'time': '11:00'}
                assert voice.task.readback_version is None and not voice.task.confirmed
                await voice.playback_finished(greeting)
                await asyncio.sleep(0)
                assert voice.task.readback_version is None
                await voice.playback_finished(fresh)
                await asyncio.wait_for(voice.playback_task, 2)
                assert voice.task.readback_version == voice.task.version
                await speak('yes', response(is_affirmation=True))
                await acknowledge()
                assert voice.task.confirmed and voice.done.is_set()
                assert text not in json.dumps(requests[-1]['messages'])
                assert not any(event['type'] == 'error' for event in events)
                for event in events:
                    if event['type'] == 'state' and event.get('pending_proposal'):
                        assert event['slots'] == original
            finally:
                await voice.close()
    asyncio.run(run())
