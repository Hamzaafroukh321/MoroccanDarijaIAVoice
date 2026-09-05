"""Router diagnostics identify static local validation phases without echoing output."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.router import Router, RouterError, RouterOutputError


def payload(**changes):
    result = dict(intent='task', ops=[], confidence=.99, unclear=False,
        is_affirmation=False, is_negation=False, clarification=None,
        proposed_ops=[], resolves_clarification=None, discard_clarification=None,
        discard_request=None)
    result.update(changes)
    return result


def operation(slot, value):
    return dict(op='set', slot=slot, value=value)


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    result = demo_config(load_config(ROOT / 'configs/clinic.json'))
    result['router']['retries'] = 0
    return result


def route(config, tmp_path, responses, *, transcript='الطبيب باع', state=None):
    async def run():
        contexts = deepcopy(state if state is not None else {'state': {}})
        original, requests = deepcopy(contexts), []
        def serve(request):
            requests.append(request)
            response = responses[min(len(requests) - 1, len(responses) - 1)]
            if isinstance(response, Exception):
                raise response
            return httpx.Response(response[0], json=response[1])
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            router = Router(config, tmp_path, api_key='offline-fixture', client=client)
            router.limiter.reserve = lambda: None
            result, error = None, None
            try:
                result = await router.route(transcript, contexts)
            except RouterError as exc:
                error = exc
            assert contexts == original
            return result, error, router.calls, len(requests)
    return asyncio.run(run())


def completion(value, **changes):
    choice = {'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}
    choice.update(changes)
    return 200, {'choices': [choice]}


@pytest.mark.parametrize('phase', ['response_completion', 'response_schema',
    'current_utterance_time', 'explicit_enum_alternatives', 'clarification_identity',
    'retained_request_discard'])
def test_local_failure_reports_the_specific_last_validator(config, tmp_path, phase):
    text, state, candidate = 'fixture utterance', {'state': {}}, payload()
    finish = 'stop'
    if phase == 'response_completion':
        finish = 'length'
    elif phase == 'response_schema':
        candidate['confidence'] = 'UNTRUSTED_OUTPUT_MARKER https://private.invalid/?secret=fixture'
    elif phase == 'current_utterance_time':
        text = 'at 11'
        candidate['ops'] = [operation('time', '08:00')]
    elif phase == 'explicit_enum_alternatives':
        text = 'doctor A or doctor B'
        candidate['ops'] = [operation('doctor', 'doctor_a')]
    elif phase == 'clarification_identity':
        state['pending_clarification'] = {'id': 7, 'kind': 'ambiguous_value', 'slot': 'doctor', 'item_ids': []}
        candidate['ops'] = [operation('doctor', 'doctor_b')]
        candidate['resolves_clarification'] = 6
    else:
        state['pending_request'] = {'id': 'request_7', 'text': 'fixture history', 'clarification_resolved': True}
        candidate['discard_request'] = 'UNTRUSTED_REQUEST_MARKER https://private.invalid/?secret=fixture'
    result, error, calls, count = route(config, tmp_path,
        [completion(candidate, finish_reason=finish)], transcript=text, state=state)
    assert result is None and isinstance(error, RouterOutputError)
    assert count == 1 and len(calls) == 1
    call = calls[0]
    assert call['failure_kind'] == 'model_output' and not call['ok']
    assert call['validation_stage'] == phase
    assert 'UNTRUSTED' not in call['validation_stage'] and 'https://' not in call['validation_stage']


@pytest.mark.parametrize('failure', ['http', 'envelope', 'timeout'])
def test_provider_or_envelope_failure_has_no_local_validation_stage(config, tmp_path, failure):
    if failure == 'http':
        reply = (503, {'error': 'UNTRUSTED_SERVICE_MARKER https://private.invalid/?secret=fixture'})
    elif failure == 'envelope':
        reply = (200, {'choices': [{'message': {}}]})
    else:
        reply = httpx.ReadTimeout('UNTRUSTED_EXCEPTION_MARKER https://private.invalid/?secret=fixture')
    result, error, calls, count = route(config, tmp_path, [reply])
    assert result is None and isinstance(error, RouterError) and not isinstance(error, RouterOutputError)
    assert count == 1 and calls[0]['failure_kind'] == 'provider'
    assert 'validation_stage' not in calls[0]
    assert 'UNTRUSTED' not in json.dumps(calls[0])


def test_retry_does_not_leak_previous_local_stage_into_provider_failure(config, tmp_path):
    config['router']['retries'] = 1
    state = {'state': {}, 'pending_clarification': {
        'id': 4, 'kind': 'ambiguous_value', 'slot': 'doctor', 'item_ids': []}}
    candidate = payload(ops=[operation('doctor', 'doctor_b')], resolves_clarification=3)
    result, error, calls, count = route(config, tmp_path,
        [completion(candidate), (503, {'error': 'offline service failure'})], state=state)
    assert result is None and isinstance(error, RouterError) and not isinstance(error, RouterOutputError)
    assert count == 2
    assert calls[0]['validation_stage'] == 'clarification_identity'
    assert 'validation_stage' not in calls[1] and calls[1]['failure_kind'] == 'provider'


def test_retry_success_has_no_failure_stage_and_preserves_existing_rule(config, tmp_path):
    config['router']['retries'] = 1
    result, error, calls, count = route(config, tmp_path, [
        completion(payload(ops=[operation('time', '08:00')])),
        completion(payload(ops=[operation('time', '11:00')]))], transcript='at 11')
    assert error is None and result.ops[0].value == '11:00' and count == 2
    assert calls[0]['validation_stage'] == 'current_utterance_time'
    assert calls[0]['validation_rule'] == 'current_utterance_time'
    assert calls[1]['ok'] and 'validation_stage' not in calls[1]
