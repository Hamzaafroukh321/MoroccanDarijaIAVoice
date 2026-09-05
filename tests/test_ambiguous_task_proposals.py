"""Explicit clear-field proposals alongside an ambiguous flat-task choice."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config, reply_text
from engine.demo_order import DemoOrderState
from engine.endpointing import Segment
from engine.pipeline import VoiceSession
from engine.responder import RenderedAudio
from engine.router import Router, RouterOutputError, parse_response
from engine.scoped_task import ScopedTaskState
from engine.stt import Transcript, wav_bytes


def payload(**changes):
    result = dict(intent='task', ops=[], confidence=.99, unclear=False,
        is_affirmation=False, is_negation=False, clarification=None, proposed_ops=[],
        resolves_clarification=None, discard_clarification=None, discard_request=None)
    result.update(changes)
    return result


def operation(slot, value):
    return {'op': 'set', 'slot': slot, 'value': value}


@pytest.fixture(params=['clinic', 'renamed_flat'])
def case(request, monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    config = demo_config(load_config(ROOT / 'configs/clinic.json'))
    field = 'doctor'
    if request.param == 'renamed_flat':
        # A synthetic alternate configured field, without a new shipped domain.
        config = json.loads(json.dumps(config).replace('doctor', 'counter'))
        config['domain_id'] = 'service-desk'
        field = 'counter'
    proposal = payload(intent='ambiguous', clarification={
        'kind': 'ambiguous_value', 'slot': field, 'item_ids': []}, proposed_ops=[
        operation('date', '2026-09-15'), operation('time', '11:30')])
    return dict(config=config, field=field, proposal=proposal)


def consume(state, body, config):
    return state.consume(parse_response(json.dumps(body), config).model_dump())


def snapshot(state):
    return deepcopy((state.values, state.version, state.pending_proposal,
                     state.pending_clarification, state.repair_count, state.readback_version))


def test_clear_fields_wait_for_explicit_resolution_and_fresh_confirmation(case):
    config, field = case['config'], case['field']
    state = ScopedTaskState(config)
    assert consume(state, case['proposal'], config).kind == 'ambiguous_value'
    assert state.values == {} and state.version == 0
    pending = state.pending_clarification['id']
    held = deepcopy(state.pending_proposal)
    assert held['state'] == {'date': '2026-09-15', 'time': '11:30'}
    assert consume(state, payload(is_affirmation=True), config).kind == 'ambiguous_value'
    assert state.pending_proposal == held and state.values == {} and not state.confirmed
    state.begin_confirmation(state.version)
    assert state.readback_version is None
    resolution = payload(ops=[operation(field, field + '_b')], resolves_clarification=pending)
    assert consume(state, resolution, config).kind == 'readback'
    assert state.values == {'date': '2026-09-15', 'time': '11:30', field: field + '_b'}
    assert state.version == 1 and state.pending_proposal is None and state.pending_clarification is None
    committed = snapshot(state)
    with pytest.raises(ValueError):
        consume(state, resolution, config)
    assert snapshot(state) == committed
    assert consume(state, payload(is_affirmation=True), config).kind == 'readback'
    assert not state.confirmed
    state.begin_confirmation(state.version)
    assert consume(state, payload(is_affirmation=True), config).kind == 'accepted'
    assert state.confirmed


def test_discard_ambiguous_proposal_preserves_committed_facts_and_invalidates_ack(case):
    config, field = case['config'], case['field']
    state = ScopedTaskState(config)
    original = {field: field + '_a', 'date': '2026-09-14', 'time': '10:00'}
    state.apply([operation(key, value) for key, value in original.items()])
    old_version = state.version
    state.begin_confirmation(old_version)
    consume(state, case['proposal'], config)
    pending = state.pending_clarification['id']
    assert state.values == original and state.readback_version is None
    assert consume(state, payload(discard_clarification=pending), config).kind == 'readback'
    assert state.values == original and state.version > old_version
    assert state.pending_proposal is None and state.pending_clarification is None
    state.begin_confirmation(old_version)
    assert state.readback_version is None and not state.confirmed
    with pytest.raises(ValueError):
        consume(state, payload(discard_clarification=pending), config)


@pytest.mark.parametrize('failure', ['same_field', 'unclear', 'wrong_intent', 'mixed_task', 'invalid_value'])
def test_invalid_ambiguous_proposal_is_atomic(case, failure):
    config, field = case['config'], case['field']
    state = ScopedTaskState(config)
    state.apply([operation('time', '10:00')])
    body = deepcopy(case['proposal'])
    if failure == 'same_field':
        body['proposed_ops'].append(operation(field, field + '_a'))
    elif failure == 'unclear':
        body['unclear'] = True
    elif failure == 'wrong_intent':
        body['intent'] = 'out_of_scope'
    elif failure == 'mixed_task':
        body.update(intent='task', ops=body['proposed_ops'], proposed_ops=[])
    elif failure == 'invalid_value':
        body['proposed_ops'].append(operation('time', '99:88'))
    before = snapshot(state)
    with pytest.raises(ValueError):
        consume(state, body, config)
    assert snapshot(state) == before


def test_active_question_cannot_be_replaced_with_another_proposal(case):
    state = ScopedTaskState(case['config'])
    consume(state, case['proposal'], case['config'])
    before = snapshot(state)
    replacement = payload(intent='ambiguous', clarification={
        'kind': 'ambiguous_value', 'slot': 'time', 'item_ids': []},
        proposed_ops=[operation('date', '2026-10-01')])
    with pytest.raises(ValueError):
        consume(state, replacement, case['config'])
    assert snapshot(state) == before


@pytest.mark.parametrize('finish', ['resolve', 'discard'])
def test_pipeline_retains_original_until_resolution_or_discard_without_partial_commit(case, tmp_path, finish):
    config, field = case['config'], case['field']
    async def run():
        events, contexts = [], []
        class ASR:
            turn = 0
            async def transcribe(self, pcm):
                self.turn += 1
                return Transcript(f'fictional request {self.turn}', .99, 'offline')
        class Route:
            async def route(self, text, context):
                contexts.append(deepcopy(context))
                turn = len(contexts)
                if turn == 1:
                    result = case['proposal']
                elif turn == 2:
                    result = payload(is_affirmation=True)
                elif turn == 3:
                    pending = context['pending_clarification']['id']
                    result = (payload(ops=[operation(field, field + '_b')], resolves_clarification=pending)
                              if finish == 'resolve' else payload(discard_clarification=pending))
                else:
                    assert turn == 4 and context['pending_request'] is None
                    result = payload(is_affirmation=True) if finish == 'resolve' else payload(intent='greeting')
                return parse_response(json.dumps(result), config)
        class Bank:
            async def render(self, action, state):
                return RenderedAudio(wav_bytes(bytes(2048), config['runtime']), 1024, 1024,
                    ['offline'], reply_text(action, state, config['demo']))
        async def emit(event):
            events.append(event)
            if event['type'] == 'audio_end':
                await session.playback_finished(event['playback_id'])
        async def audio(data):
            pass
        session = VoiceSession(config, tmp_path, ASR(), Route(), Bank(), None, emit, audio)
        try:
            async with asyncio.timeout(3):
                await session.start()
                await session.playback_task
                for turn in range(1, 5):
                    await session.queue.put(Segment(bytes([turn, 0]) * 512, 0, 320, 1728, 320))
                    await session.queue.join()
                    await session.playback_task
                    if turn < 3:
                        assert session.task.values == {} and not session.task.confirmed
                        assert session.pending_request['text'] == 'fictional request 1'
                        assert session.task.readback_version is None
                    if turn == 3:
                        assert session.pending_request is None
                        assert not session.task.confirmed
                assert contexts[1]['pending_request'] == contexts[2]['pending_request']
                assert contexts[1]['pending_proposal'] == contexts[2]['pending_proposal']
                if finish == 'resolve':
                    assert session.task.values == {field: field + '_b', 'date': '2026-09-15', 'time': '11:30'}
                    assert session.task.version == 1 and session.task.confirmed
                    assert session.status == 'demo_completed'
                else:
                    assert session.task.values == {} and not session.task.confirmed
                assert not any(event['type'] in {'error', 'warning'} for event in events)
        finally:
            await session.close()
        saved = json.loads(next(tmp_path.glob('bench/results/demo_session_*.json')).read_text(encoding='utf-8'))
        assert saved['turns'][0]['state'] == saved['turns'][1]['state'] == {}
        assert saved['pending_proposal'] is None and saved['pending_clarification'] is None
    asyncio.run(run())


def test_ambiguous_time_proposals_obey_current_utterance_grounding(case, tmp_path):
    config = case['config']
    config['router']['retries'] = 1
    wrong = deepcopy(case['proposal'])
    wrong['proposed_ops'] = [operation('time', '08:00')]
    right = deepcopy(wrong)
    right['proposed_ops'] = [operation('time', '11:00')]
    async def run():
        requests = []
        def handle(request):
            requests.append(request)
            result = wrong if len(requests) == 1 else right
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                'message': {'content': json.dumps(result)}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            router = Router(config, tmp_path, api_key='offline', client=client)
            router.limiter.reserve = lambda: None
            parsed = await router.route('choice unclear at 11', {'state': {}, 'requested_slot': case['field']})
            assert len(requests) == 2
            assert router.calls[0]['validation_rule'] == 'current_utterance_time'
            assert parsed.proposed_ops[0].value == '11:00' and parsed.ops == []
    asyncio.run(run())


def test_pizza_ambiguous_proposals_remain_unsupported(monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    config = demo_config(load_config(ROOT / 'configs/pizza.json'))
    state = DemoOrderState(config)
    body = payload(intent='ambiguous', clarification={
        'kind': 'order_details', 'slot': 'quantity', 'item_ids': []}, proposed_ops=[
        {'op': 'create', 'item_id': 1, 'slot': None, 'value': None},
        {'op': 'set', 'item_id': 1, 'slot': 'size', 'value': 'large'}])
    before = snapshot(state)
    with pytest.raises(ValueError):
        consume(state, body, config)
    assert snapshot(state) == before and state.next_item_id == 1


@pytest.mark.parametrize('retry', ['valid_proposal', 'second_bad', 'pending_question'])
def test_mixed_ambiguous_candidate_gets_static_retry_without_committing_or_replacing_question(case, tmp_path, retry):
    config = case['config']
    config['router']['retries'] = 1
    valid = deepcopy(case['proposal'])
    bad = deepcopy(valid)
    bad.update(intent='task', ops=bad['proposed_ops'], proposed_ops=[])
    state = ScopedTaskState(config)
    if retry == 'pending_question':
        consume(state, case['proposal'], config)
    before = snapshot(state)
    async def run():
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            candidate = bad if len(requests) == 1 or retry == 'second_bad' else valid
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                'message': {'content': json.dumps(candidate)}}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            router = Router(config, tmp_path, api_key='offline', client=client)
            router.limiter.reserve = lambda: None
            if retry == 'second_bad':
                with pytest.raises(RouterOutputError):
                    await router.route('diagnostic request', state.router_context())
            else:
                parsed = await router.route('diagnostic request', state.router_context())
                assert snapshot(state) == before
                if retry == 'pending_question':
                    with pytest.raises(ValueError):
                        state.consume(parsed.model_dump())
                else:
                    assert parsed.intent == 'ambiguous' and parsed.ops == []
                    assert parsed.proposed_ops
                    assert not getattr(parsed, '_mixed_proposal_normalized', False)
            assert snapshot(state) == before
            assert len(requests) == len(router.calls) == 2
            failed = router.calls[0]
            assert not failed['ok'] and failed['error'] == 'ValidationError'
            assert failed['failure_kind'] == 'model_output'
            assert failed['response_shape'] == {'intent': 'task', 'ops_count': 2,
                'proposed_ops_count': 0, 'clarification_kind': 'ambiguous_value',
                'clarification_slot': case['field']}
            first, second = [request['messages'] for request in requests]
            assert second[:-1] == first
            reminder = second[-1]
            assert reminder['role'] == 'system'
            assert 'intent=ambiguous and ops=[]' in reminder['content']
            assert 'initial question' in reminder['content']
            for rejected_data in ('2026-09-15', '11:30', case['field'], json.dumps(bad)):
                assert rejected_data not in reminder['content']
            assert not any(message['role'] == 'assistant' for message in second)
            if retry == 'second_bad':
                assert all(not call['ok'] for call in router.calls)
            else:
                assert router.calls[1]['ok']
    asyncio.run(run())
