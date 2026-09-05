"""Offline HTTP-200 router validation failures must permit bounded repair."""

import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config, reply_text
from engine.endpointing import Segment
from engine.pipeline import VoiceSession
from engine.responder import RenderedAudio
from engine.router import Router
from engine.stt import Transcript, wav_bytes


def response(**changes):
    result = dict(intent='task', ops=[], confidence=.99, unclear=False,
                  is_affirmation=False, is_negation=False, clarification=None,
                  resolves_clarification=None, proposed_ops=[], discard_clarification=None)
    result.update(changes)
    return result


def operation(domain, slot, value, item_id=1):
    result = dict(op='set', slot=slot, value=value)
    if domain == 'pizza':
        result['item_id'] = None if slot == 'drink' else item_id
    return result


def complete_operations(domain):
    if domain == 'clinic':
        return [operation(domain, key, value) for key, value in
                {'doctor': 'doctor_a', 'date': '2026-09-15', 'time': '10:30'}.items()]
    return [dict(op='create', item_id=1, slot=None, value=None),
            *[operation(domain, key, value) for key, value in
              {'quantity': 1, 'size': 'large', 'toppings': ['cheese'], 'drink': ['cola']}.items()]]


class SessionFixture:
    def __init__(self, domain, root):
        self.domain = domain
        self.config = demo_config(load_config(ROOT / 'configs' / f'{domain}.json'))
        self.config['router']['retries'] = 1
        self.config['demo']['max_repairs'] = 2
        self.text = 'fixture request'
        self.payload = response()
        self.failure = None
        self.requests = []
        self.events = []
        self.turn = 0
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self.http))
        self.router = Router(self.config, root, api_key='offline-fixture', client=self.client)
        self.router.limiter.reserve = lambda: None
        self.voice = VoiceSession(self.config, root, self, self.router, self, None,
                                  self.emit, self.audio)

    def http(self, request):
        self.requests.append(json.loads(request.content))
        failure = self.failure.pop(0) if isinstance(self.failure, list) else self.failure
        if failure == 'timeout':
            raise httpx.ReadTimeout('offline timeout', request=request)
        if failure == 'http':
            return httpx.Response(503, json={'error': 'offline unavailable'})
        if failure == 'envelope':
            return httpx.Response(200, json={'choices': [{'message': {}}]})
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps(self.payload)}}]})

    async def transcribe(self, pcm):
        return Transcript(self.text, .99, 'offline-fixture')

    async def render(self, action, values):
        return RenderedAudio(wav_bytes(bytes(1024), self.config['runtime']), 512, 512,
                             ['fixture'], reply_text(action, values, self.config['demo']))

    async def emit(self, event):
        self.events.append(deepcopy(event))
        if event['type'] == 'audio_end':
            await self.voice.playback_finished(event['playback_id'])

    async def audio(self, data):
        pass

    async def start(self):
        await self.voice.start()
        await asyncio.wait_for(self.voice.playback_task, 1)

    async def speak(self, payload, text='fixture request'):
        self.payload, self.text = payload, text
        self.turn += 1
        await self.voice.queue.put(Segment(bytes([self.turn, 0]) * 512, 0, 320, 1728, 320))
        await asyncio.wait_for(self.voice.queue.join(), 1)
        if self.voice.playback_task:
            await asyncio.wait_for(self.voice.playback_task, 1)

    async def close(self):
        await self.voice.close()
        await self.client.aclose()


def incoherent(domain):
    # Includes a valid edit that must not be applied when the response as a
    # whole violates local confirmation coherence validation.
    slot, value = ('size', 'small') if domain == 'pizza' else ('doctor', 'doctor_b')
    return response(ops=[operation(domain, slot, value)],
                    is_affirmation=True, is_negation=True)


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
@pytest.mark.parametrize('invalid_kind', ['coherence', 'slot_value'])
def test_exhausted_output_validation_preserves_state_and_allows_correction(tmp_path, domain, invalid_kind):
    async def run():
        fixture = SessionFixture(domain, tmp_path)
        voice = fixture.voice
        try:
            await fixture.start()
            await fixture.speak(response(ops=complete_operations(domain)))
            assert voice.task.readback_version == voice.task.version
            old_state, old_version = deepcopy(voice.task.values), voice.task.version
            before = len(fixture.requests)
            invalid = incoherent(domain)
            if invalid_kind == 'slot_value':
                invalid = response(ops=[operation(domain,
                    'size' if domain == 'pizza' else 'doctor', 'unconfigured_choice')])
            await fixture.speak(invalid, 'correction needing a repeat')
            assert len(fixture.requests) - before == fixture.config['router']['retries'] + 1
            assert all(not call['provider_failure'] for call in fixture.router.calls[-2:])
            assert all(call['failure_kind'] == 'model_output' for call in fixture.router.calls[-2:])
            assert voice.status == 'in_progress' and not voice.done.is_set()
            assert voice.task.values == old_state and voice.task.version == old_version
            assert voice.task.readback_version is None and not voice.task.confirmed
            assert not any(event['type'] == 'error' for event in fixture.events)
            assert voice.task.repair_count == 1
            correction = incoherent(domain)['ops']
            await fixture.speak(response(ops=correction), 'clear correction')
            assert voice.task.values != old_state and voice.task.repair_count == 0
            assert voice.task.readback_version == voice.task.version and not voice.task.confirmed
            await fixture.speak(response(is_affirmation=True), 'yes')
            assert voice.task.confirmed and voice.status == 'demo_completed'
        finally:
            await fixture.close()
    asyncio.run(run())


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_invalid_output_does_not_replace_or_commit_pending_proposal(tmp_path, domain):
    async def run():
        fixture = SessionFixture(domain, tmp_path)
        voice = fixture.voice
        try:
            await fixture.start()
            pending_slot = 'drink' if domain == 'pizza' else 'doctor'
            proposed = [op for op in complete_operations(domain) if op['slot'] != pending_slot]
            kind = 'drink_size' if domain == 'pizza' else 'unsupported_value'
            await fixture.speak(response(intent='out_of_scope', proposed_ops=proposed,
                clarification=dict(kind=kind, slot=pending_slot, item_ids=[])), 'original request')
            original = deepcopy((voice.task.values, voice.task.pending_proposal,
                                 voice.task.pending_clarification, voice.pending_request))
            await fixture.speak(incoherent(domain), 'failed answer')
            assert voice.status == 'in_progress' and not voice.done.is_set()
            assert (voice.task.values, voice.task.pending_proposal,
                    voice.task.pending_clarification, voice.pending_request) == original
            assert voice.task.readback_version is None and not voice.task.confirmed
            value = ['cola'] if domain == 'pizza' else 'doctor_a'
            await fixture.speak(response(ops=[operation(domain, pending_slot, value)],
                resolves_clarification=voice.task.pending_clarification['id']), 'supported answer')
            assert voice.task.pending_proposal is None and voice.pending_request is None
            assert voice.task.ready and not voice.task.confirmed
            assert voice.task.readback_version == voice.task.version
            if domain == 'pizza':
                assert len(voice.task.values['items']) == 1 and voice.task.next_item_id == 2
        finally:
            await fixture.close()
    asyncio.run(run())


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
@pytest.mark.parametrize('active_input', [False, True])
def test_repeated_invalid_output_has_bounded_handoff(tmp_path, domain, active_input):
    async def run():
        fixture = SessionFixture(domain, tmp_path)
        voice = fixture.voice
        try:
            await fixture.start()
            old_state = deepcopy(voice.task.values)
            for index in range(fixture.config['demo']['max_repairs'] + 1):
                if active_input and index == fixture.config['demo']['max_repairs']:
                    voice.detector.buffer.extend(bytes(1024))
                await fixture.speak(incoherent(domain))
                assert voice.task.values == old_state and not voice.task.confirmed
                if index < fixture.config['demo']['max_repairs']:
                    assert voice.status == 'in_progress' and not voice.done.is_set()
            assert voice.status == 'handoff' and voice.done.is_set()
            if active_input:
                assert voice.detector.active
            assert not any(event['type'] == 'error' for event in fixture.events)
            assert len(fixture.requests) == 6
        finally:
            await fixture.close()
    asyncio.run(run())


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
@pytest.mark.parametrize('failures', [['envelope', 'envelope'], ['http', None], [None, 'timeout']])
def test_protocol_or_mixed_failures_are_not_misclassified_as_bad_speech(tmp_path, domain, failures):
    async def run():
        fixture = SessionFixture(domain, tmp_path)
        try:
            await fixture.start()
            fixture.failure = list(failures)
            await fixture.speak(incoherent(domain))
            assert fixture.voice.status == 'error' and fixture.voice.done.is_set()
            assert fixture.voice.task.repair_count == 0 and not fixture.voice.task.confirmed
            assert any(event['type'] == 'error' for event in fixture.events)
            assert any(call['failure_kind'] == 'provider' for call in fixture.router.calls)
            assert fixture.voice.turns[-1]['processing_error']['type'] == 'RouterError'
        finally:
            await fixture.close()
    asyncio.run(run())


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
@pytest.mark.parametrize('failure', ['http', 'timeout'])
def test_actual_provider_failures_remain_terminal(tmp_path, domain, failure):
    async def run():
        fixture = SessionFixture(domain, tmp_path)
        try:
            await fixture.start()
            fixture.failure = failure
            await fixture.speak(response())
            assert fixture.voice.status == 'error' and fixture.voice.done.is_set()
            assert fixture.voice.task.repair_count == 0 and not fixture.voice.task.confirmed
            assert any(event['type'] == 'error' for event in fixture.events)
            assert all(call['provider_failure'] for call in fixture.router.calls)
        finally:
            await fixture.close()
    asyncio.run(run())
