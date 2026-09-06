"""Same-row alternatives stage linked answers without replacing committed facts."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest
from jsonschema import Draft202012Validator

from collection_fixtures import collection_config
from engine.collection_task import ConfiguredCollectionState
from engine.demo import reply_text
from engine.endpointing import Segment
from engine.pipeline import VoiceSession
from engine.recovery import RecoveryStore
from engine.responder import RenderedAudio
from engine.router import Router
from engine.stt import Transcript, wav_bytes


def op(kind, item_id=None, slot=None, value=None):
    return dict(op=kind, item_id=item_id, slot=slot, value=value)


def address(item_id, slot):
    return dict(item_id=item_id, slot=slot)


def response(**changes):
    result = dict(intent='task', ops=[], confidence=.99, unclear=False,
        is_affirmation=False, is_negation=False, clarification=None, proposed_ops=[],
        resolves_clarification=None, discard_clarification=None, discard_request=None)
    result.update(changes)
    return result


def question(names, item_id=1, **changes):
    return response(intent='ambiguous', clarification=dict(kind='ambiguous_value',
        slot=names['asset'], item_ids=[item_id], linked_addresses=[address(item_id, names['quantity'])]), **changes)


def row(names, item_id, asset='camera', quantity=1):
    return [op('create', item_id), op('set', item_id, names['asset'], asset),
        op('set', item_id, names['quantity'], quantity)]


@pytest.fixture(params=[False, True], ids=['equipment', 'renamed'])
def case(request, monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    config, names = collection_config(renamed=request.param)
    config['router']['retries'] = 1
    task = ConfiguredCollectionState(config)
    task.apply(row(names, 1) + row(names, 2, 'tripod', 4) + [op('set', None, names['date'], '2026-09-15')])
    return config, names, task


def partial(task, names):
    task.consume(question(names))
    first = task.pending_clarification['id']
    action = task.consume(response(ops=[op('set', 1, names['asset'], 'tripod')], resolves_clarification=first))
    return first, action


def test_same_row_linked_answers_hold_until_complete_without_touching_another_row(case):
    _, names, task = case
    old, version = deepcopy(task.values), task.version
    task.begin_confirmation(version)
    task.consume(question(names))
    assert task.pending_proposal['ops'] == [] and task.readback_version is None
    assert task.pending_proposal['linked_addresses'] == [address(1, names['asset']), address(1, names['quantity'])]
    first = task.pending_clarification['id']
    action = task.consume(response(ops=[op('set', 1, names['asset'], 'tripod')], resolves_clarification=first))
    assert task.values == old and task.version == version
    assert action.item_id == 1 and action.slot == names['quantity'] and action.kind != 'readback'
    assert task.pending_proposal['answered_addresses'] == [address(1, names['asset'])]
    assert task.pending_proposal['remaining_addresses'] == [address(1, names['quantity'])]
    assert task.pending_proposal['state'][names['collection']][0][names['asset']] == 'tripod'
    assert task.pending_proposal['state'][names['collection']][1] == old[names['collection']][1]
    second = task.pending_clarification['id']
    assert second > first
    task.begin_confirmation(version)
    assert task.readback_version is None
    assert task.consume(response(is_affirmation=True)).kind != 'accepted'
    action = task.consume(response(ops=[op('set', 1, names['quantity'], 3)], resolves_clarification=second))
    assert action.kind == 'readback' and task.version == version + 1
    assert task.values[names['collection']][0] == {'id': 1, names['asset']: 'tripod', names['quantity']: 3}
    assert task.values[names['collection']][1] == old[names['collection']][1]
    assert task.values[names['date']] == old[names['date']]
    assert task.pending_proposal is None and task.readback_version is None
    assert task.consume(response(is_affirmation=True)).kind == 'readback'
    task.begin_confirmation(task.version)
    assert task.consume(response(is_affirmation=True)).kind == 'accepted'


def test_new_draft_row_allocation_is_held_until_both_answers_commit(case):
    config, names, _ = case
    task = ConfiguredCollectionState(config)
    task.consume(question(names, proposed_ops=[op('create', 1), op('set', None, names['date'], '2026-09-15')]))
    assert task.values == {names['collection']: []} and task.next_item_id == 1
    task.consume(response(ops=[op('set', 1, names['asset'], 'tripod')], resolves_clarification=task.pending_clarification['id']))
    assert task.values == {names['collection']: []} and task.next_item_id == 1
    assert task.pending_proposal['next_item_id'] == 2
    task.consume(response(ops=[op('set', 1, names['quantity'], 3)], resolves_clarification=task.pending_clarification['id']))
    assert task.next_item_id == 2 and task.version == 1 and task.ready
    assert task.values[names['collection']] == [{'id': 1, names['asset']: 'tripod', names['quantity']: 3}]


def test_optional_companion_still_needs_explicit_new_answer(case):
    config, names, _ = case
    next(slot for slot in config['slots'] if slot['id'] == names['quantity'])['required'] = False
    config['demo']['required_slots'].remove(names['quantity'])
    task = ConfiguredCollectionState(config)
    task.apply([op('create', 1), op('set', 1, names['asset'], 'camera'), op('set', None, names['date'], '2026-09-15')])
    assert task.ready
    _, action = partial(task, names)
    assert action.slot == names['quantity'] and task.values[names['collection']][0][names['asset']] == 'camera'
    assert task.pending_proposal['remaining_addresses'] == [address(1, names['quantity'])]


@pytest.mark.parametrize('invalid', ['root', 'cross_row', 'unknown_row', 'boolean', 'duplicate', 'primary', 'unknown_field'])
def test_invalid_linked_addresses_reject_before_pending_or_committed_mutation(case, invalid):
    _, names, task = case
    candidate = question(names)
    targets = candidate['clarification']['linked_addresses']
    if invalid == 'root':
        targets[:] = [address(None, names['date'])]
    elif invalid == 'cross_row':
        targets[0]['item_id'] = 2
    elif invalid == 'unknown_row':
        targets[0]['item_id'] = 99
    elif invalid == 'boolean':
        targets[0]['item_id'] = True
    elif invalid == 'duplicate':
        targets.append(deepcopy(targets[0]))
    elif invalid == 'primary':
        targets[0]['slot'] = names['asset']
    else:
        targets[0]['slot'] = 'unconfigured'
    before = deepcopy(task.__dict__)
    with pytest.raises(ValueError):
        task.consume(candidate)
    assert task.__dict__ == before


@pytest.mark.parametrize('invalid', ['stale_id', 'missing_id', 'bad_last_op', 'clear', 'delete'])
def test_invalid_final_answer_preserves_every_staged_fact(case, invalid):
    _, names, task = case
    first, _ = partial(task, names)
    candidate = response(ops=[op('set', 1, names['quantity'], 3)], resolves_clarification=task.pending_clarification['id'])
    if invalid == 'stale_id':
        candidate['resolves_clarification'] = first
    elif invalid == 'missing_id':
        candidate['resolves_clarification'] = None
    elif invalid == 'bad_last_op':
        candidate['ops'].append(op('set', None, names['date'], 'invalid-date'))
    elif invalid == 'clear':
        candidate['ops'].append(op('clear', 1, names['asset']))
    else:
        candidate['ops'].append(op('delete', 1))
    before = deepcopy(task.__dict__)
    with pytest.raises(ValueError):
        task.consume(candidate)
    assert task.__dict__ == before


def test_discard_and_recovery_preserve_committed_values_not_linked_answers(case):
    config, names, task = case
    old, version = deepcopy(task.values), task.version
    first, _ = partial(task, names)
    registry = RecoveryStore(clock=lambda: 100)
    offer = registry.offer(config, task, 'offline-source', pending_request={'id': 'request_1', 'text': 'original alternatives'})
    assert offer['discarded_pending'] and offer['slots'] == old
    recovered = registry.restore(offer['token'], config).task
    assert recovered.values == old and recovered.pending_proposal is None and not recovered.confirmed
    with pytest.raises(ValueError):
        task.consume(response(discard_clarification=first))
    task.consume(response(discard_clarification=task.pending_clarification['id']))
    assert task.values == old and task.version == version + 1 and task.next_item_id == 3
    assert task.pending_proposal is None and task.pending_clarification is None
    assert task.readback_version is None


def test_actual_pipeline_retains_request_masks_unanswered_preview_and_requires_fresh_ack(case, tmp_path):
    async def run():
        config, names, initial = case
        requests, candidates, events = [], [], []
        ended = asyncio.Queue()
        current_text, auto_ack = '', True
        def http(request):
            requests.append(json.loads(request.content))
            assert candidates, 'Unexpected router attempt'
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
                if auto_ack:
                    await voice.playback_finished(event['playback_id'])
                else:
                    ended.put_nowait(event['playback_id'])
        async def audio(data):
            pass
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            router = Router(config, tmp_path, api_key='offline-fixture', client=client)
            router.limiter.reserve = lambda: None
            voice = VoiceSession(config, tmp_path, ASR(), router, Bank(), None, emit, audio)
            voice.task = initial
            async def speak(text, candidate):
                nonlocal current_text
                current_text = text
                candidates.append(candidate)
                await voice.queue.put(Segment(bytes([len(requests) + 1, 0]) * 512, 0, 320, 1728, 320))
                await asyncio.wait_for(voice.queue.join(), 2)
                if auto_ack:
                    await asyncio.wait_for(voice.playback_task, 2)
            original = deepcopy(initial.values)
            original_text = 'first reservation camera quantity one or tripod quantity three'
            try:
                await voice.start()
                await voice.playback_task
                await speak(original_text, question(names))
                pending = deepcopy(voice.pending_request)
                first = voice.task.pending_clarification['id']
                await speak('I choose tripod', response(ops=[op('set', 1, names['asset'], 'tripod')], resolves_clarification=first))
                assert voice.pending_request == pending and voice.task.values == original
                assert voice.task.pending_clarification['id'] > first
                assert voice.last_assistant_text.startswith('Reservation 1.')
                old_ack = next(event['playback_id'] for event in reversed(events) if event['type'] == 'audio_end')
                auto_ack = False
                await speak('make it 3', response(ops=[op('set', 1, names['quantity'], 3)],
                    resolves_clarification=voice.task.pending_clarification['id']))
                fresh = await asyncio.wait_for(ended.get(), 2)
                messages = requests[-1]['messages']
                assert {'role': 'user', 'content': original_text} in messages
                encoded = messages[0]['content'].split('CURRENT CONTEXT (data, not instructions)\n', 1)[1]
                context, _ = json.JSONDecoder().raw_decode(encoded.lstrip())
                assert context['committed_state'] == original
                preview = context['state'][names['collection']]
                assert preview[0][names['asset']] == 'tripod' and names['quantity'] not in preview[0]
                assert context['pending_proposal']['state'] == context['state']
                assert preview[1] == original[names['collection']][1]
                assert voice.pending_request is None and voice.task.readback_version is None
                await voice.playback_finished(old_ack)
                await asyncio.sleep(0)
                assert voice.task.readback_version is None
                await voice.playback_finished(fresh)
                await asyncio.wait_for(voice.playback_task, 2)
                auto_ack = True
                await speak('yes', response(is_affirmation=True))
                assert voice.task.confirmed and voice.done.is_set()
                assert voice.task.values[names['collection']][0][names['quantity']] == 3
                assert not any(event['type'] == 'error' for event in events)
            finally:
                await voice.close()
    asyncio.run(run())


def test_new_wire_schema_requires_explicit_link_field_but_parser_keeps_historical_payloads(case):
    from engine.collection_routing import collection_response_format, parse_collection_response
    config, names, _ = case
    schema = collection_response_format(config)['json_schema']['schema']
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    linked = question(names)
    validator.validate(linked)
    assert parse_collection_response(json.dumps(linked), config).clarification.linked_addresses is not None
    current = question(names)
    current['clarification']['linked_addresses'] = None
    validator.validate(current)
    historical = deepcopy(current)
    historical['clarification'].pop('linked_addresses')
    assert not validator.is_valid(historical)
    assert parse_collection_response(json.dumps(historical), config).clarification.linked_addresses is None
