"""Recognized schema rejection is configuration failure, never a speech repair."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.endpointing import Segment
from engine.pipeline import VoiceSession
from engine.responder import RenderedAudio
from engine.router import Router, RouterConfigurationError, is_schema_configuration_error
from engine.scoped_task import ScopedTaskState
from engine.stt import Transcript, wav_bytes


UNTRUSTED = 'PRIVATE_PROVIDER_BODY https://private.invalid/?secret=fixture'


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    value = demo_config(load_config(ROOT / 'configs/clinic.json'))
    value['router']['retries'] = 1
    return value


def rejected(**changes):
    error = dict(type='invalid_request_error', param='response_format', message=UNTRUSTED)
    error.update(changes)
    return {'error': error}


def good():
    value = dict(intent='help', ops=[], confidence=.99, unclear=False,
        is_affirmation=False, is_negation=False, clarification=None, proposed_ops=[],
        resolves_clarification=None, discard_clarification=None, discard_request=None)
    return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
        'message': {'content': json.dumps(value)}}]})


async def attempt(config, root, first, context, observed):
    before = deepcopy(context)
    def handle(request):
        observed.setdefault('requests', []).append(request)
        return first if len(observed['requests']) == 1 else good()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        router = Router(config, root, api_key='offline-fixture', client=client)
        router.limiter.reserve = lambda: observed.setdefault('reservations', []).append(True)
        try:
            return await router.route('please help with this turn', context)
        finally:
            observed['calls'] = deepcopy(router.calls)
            assert context == before


@pytest.mark.parametrize('code', ['absent', None])
def test_observed_configuration_shape_stops_after_one_request_and_preserves_pending_state(config, tmp_path, code):
    data = rejected() if code == 'absent' else rejected(code=None)
    reply = httpx.Response(400, json=data)
    assert is_schema_configuration_error(reply)
    task = ScopedTaskState(config)
    task.apply([{'op': 'set', 'slot': 'doctor', 'value': 'doctor_a'}])
    task.consume(dict(intent='ambiguous', ops=[], unclear=False, is_affirmation=False,
        is_negation=False, proposed_ops=[{'op': 'set', 'slot': 'date', 'value': '2026-09-15'}],
        clarification={'kind': 'ambiguous_value', 'slot': 'time', 'item_ids': []}))
    before = deepcopy(task.__dict__)
    observed = {}
    with pytest.raises(RouterConfigurationError) as error:
        asyncio.run(attempt(config, tmp_path, reply, task.router_context(), observed))
    assert len(observed['requests']) == len(observed['reservations']) == len(observed['calls']) == 1
    call = observed['calls'][0]
    assert call['failure_kind'] == 'configuration' and call['validation_stage'] == 'response_format'
    assert call['retryable'] is False and not call['ok']
    assert task.__dict__ == before
    assert UNTRUSTED not in str(error.value) and UNTRUSTED not in json.dumps(observed['calls'])


def test_configuration_detection_does_not_depend_on_provider_message_wording():
    data = rejected()
    data['error'].pop('message')
    assert is_schema_configuration_error(httpx.Response(400, json=data))
    data['error']['message'] = 'A completely different localized provider explanation.'
    assert is_schema_configuration_error(httpx.Response(400, json=data))


@pytest.mark.parametrize('variant', ['unknown_code', 'generation_code', 'generation_payload',
    'generation_null', 'top_generation', 'wrong_param', 'wrong_type', 'wrong_status', 'non_json', 'oversize', 'wrong_envelope'])
def test_other_failures_keep_existing_retry_and_can_recover(config, tmp_path, variant):
    data, status = rejected(), 400
    if variant == 'unknown_code':
        data['error']['code'] = 'some_future_code'
    elif variant == 'generation_code':
        data['error']['code'] = 'json_validate_failed'
    elif variant == 'generation_payload':
        data['error']['failed_generation'] = UNTRUSTED
    elif variant == 'generation_null':
        data['error']['failed_generation'] = None
    elif variant == 'top_generation':
        data['failed_generation'] = UNTRUSTED
    elif variant == 'wrong_param':
        data['error']['param'] = 'messages'
    elif variant == 'wrong_type':
        data['error']['type'] = 'generation_error'
    elif variant == 'wrong_status':
        status = 422
    elif variant == 'oversize':
        data['error']['message'] = UNTRUSTED * 200
    elif variant == 'wrong_envelope':
        data = {'error': [UNTRUSTED]}
    reply = httpx.Response(status, text=UNTRUSTED) if variant == 'non_json' else httpx.Response(status, json=data)
    assert not is_schema_configuration_error(reply)
    observed = {}
    parsed = asyncio.run(attempt(config, tmp_path, reply, {'state': {}}, observed))
    assert parsed.intent == 'help'
    assert len(observed['requests']) == len(observed['reservations']) == len(observed['calls']) == 2
    assert observed['calls'][0]['failure_kind'] == 'provider'
    assert observed['calls'][1]['ok']
    assert UNTRUSTED not in json.dumps(observed['calls'])


@pytest.mark.parametrize('mode', ['demo', 'research'])
def test_pipeline_reports_configuration_error_without_repairing_or_changing_saved_task(config, tmp_path, mode):
    async def run():
        events, calls, renders = [], [], []
        if mode == 'research':
            config.pop('demo')
        def handle(request):
            calls.append(request)
            return httpx.Response(400, json=rejected())
        class ASR:
            async def transcribe(self, pcm):
                return Transcript('please help with this turn', .99, 'offline-fixture')
        class Bank:
            async def render(self, action, values):
                renders.append(action.kind)
                return RenderedAudio(wav_bytes(bytes(1024), config['runtime']), 512, 512, ['fixture'], 'Offline greeting')
        async def emit(event):
            events.append(deepcopy(event))
            if event['type'] == 'audio_end':
                await voice.playback_finished(event['playback_id'])
        async def audio(data):
            pass
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            router = Router(config, tmp_path, api_key='offline-fixture', client=client)
            router.limiter.reserve = lambda: None
            voice = VoiceSession(config, tmp_path, ASR(), router, Bank(), None, emit, audio)
            voice.task.apply([{'op': 'set', 'slot': 'doctor', 'value': 'doctor_a'}])
            if mode == 'demo':
                voice.task.consume(dict(intent='ambiguous', ops=[], unclear=False,
                    is_affirmation=False, is_negation=False, clarification={
                        'kind': 'ambiguous_value', 'slot': 'time', 'item_ids': []}, proposed_ops=[]))
                voice.pending_request = {'id': 'request_1', 'text': 'earlier request'}
            before = deepcopy(voice.task.values)
            try:
                await voice.start()
                await voice.playback_task
                initial_renders = list(renders)
                before_pending = deepcopy((getattr(voice.task, 'pending_clarification', None),
                    getattr(voice.task, 'pending_proposal', None), voice.pending_request))
                repairs = getattr(voice.task, 'repair_count', None)
                await voice.queue.put(Segment(bytes([1, 0]) * 512, 0, 320, 1728, 320))
                await asyncio.wait_for(voice.queue.join(), 2)
                assert voice.status == 'error' and voice.done.is_set()
                assert voice.task.values == before and getattr(voice.task, 'repair_count', None) == repairs
                assert (getattr(voice.task, 'pending_clarification', None),
                    getattr(voice.task, 'pending_proposal', None), voice.pending_request) == before_pending
                assert renders == initial_renders
                assert not voice.task.confirmed and len(calls) == 1
                errors = [event for event in events if event['type'] == 'error']
                assert len(errors) == 1 and UNTRUSTED not in errors[0]['message']
                assert not any(event['type'] == 'warning' for event in events)
                assert voice.turns[-1]['processing_error']['type'] == 'RouterConfigurationError'
            finally:
                await voice.close()
    asyncio.run(run())
