"""Explicit cancellation of resolved-but-retained history, using offline providers."""

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
from engine.router import Router, parse_response, response_format
from engine.scoped_task import ScopedTaskState
from engine.stt import Transcript, wav_bytes


def payload(**changes):
    value = dict(intent='task', ops=[], confidence=.99, unclear=False,
                 is_affirmation=False, is_negation=False, clarification=None,
                 proposed_ops=[], resolves_clarification=None, discard_clarification=None,
                 discard_request=None)
    value.update(changes)
    return value


def config_for(domain):
    config = demo_config(load_config(ROOT / 'configs' / f'{domain}.json'))
    config['router']['retries'] = 1
    config['demo']['max_repairs'] = 5
    return config


def question(domain):
    return dict(kind='drink_size' if domain == 'pizza' else 'unsupported_value',
                slot='drink' if domain == 'pizza' else 'doctor', item_ids=[])


def answer(domain, second=False):
    if domain == 'pizza':
        return dict(op='set', item_id=None, slot='drink', value=['cola' if second else 'water'])
    return dict(op='set', slot='doctor', value='doctor_b' if second else 'doctor_a')


def complete_state(domain, config):
    state = DemoOrderState(config) if domain == 'pizza' else ScopedTaskState(config)
    if domain == 'pizza':
        operations = [dict(op='create', item_id=1, slot=None, value=None),
                      dict(op='set', item_id=1, slot='quantity', value=1),
                      dict(op='set', item_id=1, slot='size', value='large'),
                      dict(op='set', item_id=1, slot='toppings', value=['cheese'])]
    else:
        operations = [dict(op='set', slot='date', value='2026-09-15'),
                      dict(op='set', slot='time', value='10:30')]
    state.apply(operations + [answer(domain)])
    return state


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_history_cancellation_clears_future_router_messages_and_stale_id_cannot_clear_new_history(tmp_path, domain):
    async def run():
        config = config_for(domain)
        requests, events, responses = [], [], []
        current_text = ''
        turn = 0
        def http(request):
            requests.append(json.loads(request.content))
            assert responses, 'Unexpected extra router call'
            result = responses.pop(0)
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                'message': {'content': json.dumps(result)}}]})
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
                await voice.playback_finished(event['playback_id'])
        async def audio(data):
            pass
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            router = Router(config, tmp_path, api_key='offline-fixture', client=client)
            router.limiter.reserve = lambda: None
            voice = VoiceSession(config, tmp_path, ASR(), router, Bank(), None, emit, audio)
            async def speak(text, *outputs):
                nonlocal current_text, turn
                current_text, turn = text, turn + 1
                responses.extend(outputs)
                await voice.queue.put(Segment(bytes([turn, 0]) * 512, 0, 320, 1728, 320))
                await asyncio.wait_for(voice.queue.join(), 2)
                await asyncio.wait_for(voice.playback_task, 2)
                assert not responses
            try:
                await voice.start()
                await voice.playback_task
                original = 'original request alpha with extra details'
                await speak(original, payload(intent='out_of_scope', clarification=question(domain)))
                assert voice.pending_request['id'] == 'request_1'
                clarification_id = voice.task.pending_clarification['id']
                await speak('supported answer', payload(ops=[answer(domain)], resolves_clarification=clarification_id))
                assert voice.pending_request['clarification_resolved'] is True
                assert voice.task.pending_clarification is None and not voice.task.ready
                committed, version = deepcopy(voice.task.values), voice.task.version
                # The real router retries the wrong request identity, then
                # returns only the explicit current identity from the next output.
                before = len(requests)
                await speak('cancel earlier request', payload(discard_request='request_99'),
                            payload(discard_request='request_1'))
                assert len(requests) == before + 2
                assert '"id": "request_1"' in requests[-1]['messages'][0]['content']
                assert voice.pending_request is None and voice.task.values == committed
                assert voice.task.version == version + 1 and voice.task.readback_version is None
                assert not voice.task.confirmed
                await speak('help now', payload(intent='help'))
                assert original not in json.dumps(requests[-1]['messages'])
                assert voice.task.values == committed

                await speak('different request beta', payload(intent='out_of_scope', clarification=question(domain)))
                assert voice.pending_request['id'] == 'request_2'
                await speak('second supported answer', payload(ops=[answer(domain, True)],
                    resolves_clarification=voice.task.pending_clarification['id']))
                retained = deepcopy(voice.pending_request)
                committed = deepcopy(voice.task.values)
                await speak('stale cancellation', payload(discard_request='request_1'),
                            payload(discard_request='request_1'))
                assert voice.pending_request == retained and voice.task.values == committed
                assert voice.status == 'in_progress' and not voice.done.is_set()
                await speak('cancel current retained request', payload(discard_request='request_2'))
                assert voice.pending_request is None and voice.task.values == committed
                assert not any(event['type'] == 'error' for event in events)
            finally:
                await voice.close()
    asyncio.run(run())


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_shared_history_discard_preserves_completed_values_and_requires_fresh_readback(domain):
    config = config_for(domain)
    state = complete_state(domain, config)
    retained = {'id': 'request_7', 'text': 'prior request', 'truncated': False,
                'clarification_resolved': True}
    state.begin_confirmation(state.version)
    old_version, old_values = state.version, deepcopy(state.values)
    # A mixed cancellation/edit must reject before changing either task data or history.
    with pytest.raises(ValueError):
        state.consume(payload(discard_request='request_7', ops=[answer(domain, True)]), retained_request=retained)
    with pytest.raises(ValueError):
        state.consume(payload(discard_request='request_7', is_affirmation=None), retained_request=retained)
    assert state.version == old_version and state.values == old_values
    action = state.consume(payload(discard_request='request_7'), retained_request=retained)
    assert action.kind == 'readback' and state.values == old_values
    assert state.version == old_version + 1 and state.readback_version is None
    assert retained['id'] == 'request_7'  # Session owns removing retained context.
    state.begin_confirmation(old_version)
    assert state.consume(payload(is_affirmation=True)).kind == 'readback'
    assert not state.confirmed
    state.begin_confirmation(state.version)
    assert state.consume(payload(is_affirmation=True)).kind == 'accepted'


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
@pytest.mark.parametrize('phase', ['missing', 'unresolved', 'active_clarification'])
def test_history_discard_requires_current_resolved_history_with_no_active_question(domain, phase):
    config = config_for(domain)
    state = complete_state(domain, config)
    retained = {'id': 'request_4', 'text': 'retained', 'clarification_resolved': phase != 'unresolved'}
    if phase == 'missing':
        retained = None
    elif phase == 'active_clarification':
        state.consume(payload(intent='out_of_scope', clarification=question(domain)))
    before = deepcopy((state.values, state.version, state.pending_clarification, state.pending_proposal))
    with pytest.raises(ValueError):
        state.consume(payload(discard_request='request_4'), retained_request=retained)
    assert (state.values, state.version, state.pending_clarification, state.pending_proposal) == before


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_history_discard_field_is_strict_api_required_but_legacy_parser_compatible(domain):
    config = config_for(domain)
    schema = response_format(config)['json_schema']['schema']
    assert 'discard_request' in schema['required']
    old_payload = payload()
    del old_payload['discard_request']
    assert parse_response(json.dumps(old_payload), config).discard_request is None


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_new_question_replaces_resolved_history_but_active_question_keeps_original(tmp_path, domain):
    async def run():
        config = config_for(domain)
        requests, queued = [], []
        text, turn = '', 0
        def http(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                'message': {'content': json.dumps(queued.pop(0))}}]})
        class Fixtures:
            async def transcribe(self, pcm):
                return Transcript(text, .99, 'offline-fixture')
            async def render(self, action, values):
                return RenderedAudio(wav_bytes(bytes(1024), config['runtime']), 512, 512,
                    ['fixture'], reply_text(action, values, config['demo']))
        async def emit(event):
            if event['type'] == 'audio_end':
                await voice.playback_finished(event['playback_id'])
        async def audio(data):
            pass
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            router = Router(config, tmp_path, api_key='offline-fixture', client=client)
            router.limiter.reserve = lambda: None
            fixture = Fixtures()
            voice = VoiceSession(config, tmp_path, fixture, router, fixture, None, emit, audio)
            async def speak(utterance, result):
                nonlocal text, turn
                text, turn = utterance, turn + 1
                queued.append(result)
                await voice.queue.put(Segment(bytes([turn, 0]) * 512, 0, 320, 1728, 320))
                await asyncio.wait_for(voice.queue.join(), 2)
                await asyncio.wait_for(voice.playback_task, 2)
                assert not queued
            try:
                await voice.start()
                await voice.playback_task
                alpha, beta = 'original request alpha details', 'new request beta details'
                await speak(alpha, payload(intent='out_of_scope', clarification=question(domain)))
                first_question = voice.task.pending_clarification['id']
                await speak('supported first answer', payload(ops=[answer(domain)],
                    resolves_clarification=first_question))
                assert voice.pending_request['text'] == alpha
                assert voice.pending_request['clarification_resolved'] is True
                committed = deepcopy(voice.task.values)

                await speak(beta, payload(intent='out_of_scope', clarification=question(domain)))
                active = deepcopy(voice.task.pending_clarification)
                retained = deepcopy(voice.pending_request)
                assert active['id'] > first_question
                assert retained['id'] == 'request_2' and retained['text'] == beta
                assert not retained.get('clarification_resolved', False)
                assert voice.task.values == committed

                # A new question was created above. This later unresolved turn
                # does not create one: keep its question and request identities.
                later_question = (dict(kind='unsupported_drink', slot='drink', item_ids=[])
                                  if domain == 'pizza' else
                                  dict(kind='ambiguous_value', slot='doctor', item_ids=[]))
                await speak('later failed question attempt', payload(
                    intent='out_of_scope' if domain == 'pizza' else 'ambiguous',
                    clarification=later_question))
                assert voice.task.pending_clarification == active and voice.pending_request == retained
                messages = json.dumps(requests[-1]['messages'])
                assert beta in messages and alpha not in messages
                assert '"id": "request_2"' in requests[-1]['messages'][0]['content']

                await speak('cancel active question', payload(discard_clarification=active['id']))
                assert voice.pending_request is None and voice.task.pending_clarification is None
                assert voice.task.values == committed and not voice.task.confirmed
                await speak('help after cancellation', payload(intent='help'))
                messages = json.dumps(requests[-1]['messages'])
                assert beta not in messages and alpha not in messages
            finally:
                await voice.close()
    asyncio.run(run())
