"""Collection rejection provenance is useful without retaining rejected input."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest
from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from collection_fixtures import collection_config
from test_collection_linked import address, op, response, row
from engine.collection_routing import parse_collection_response
from engine.collection_task import ConfiguredCollectionState
from engine.collection_validation import CollectionValidationError, VALIDATION_RULES, validation_error_rules
from engine.endpointing import Segment
from engine.pipeline import VoiceSession
from engine.responder import RenderedAudio
from engine.router import Router, RouterOutputError
from engine.stt import Transcript, wav_bytes


SECRET = 'UNTRUSTED_MODEL_SENTINEL_https://private.invalid/?key=fixture'


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    return collection_config()


def question(names, **changes):
    scope = dict(kind='ambiguous_value', slot=names['asset'], item_ids=[1], linked_addresses=None)
    scope.update(changes)
    return response(intent='ambiguous', clarification=scope)


def reject(raw, config, expected):
    with pytest.raises(CollectionValidationError) as failure:
        parse_collection_response(raw if isinstance(raw, str) else json.dumps(raw), config)
    error = failure.value
    assert expected in error.validation_rules
    assert isinstance(error.validation_rules, tuple)
    assert set(error.validation_rules) <= VALIDATION_RULES
    assert list(error.validation_rules) == sorted(set(error.validation_rules))
    assert len(error.validation_rules) <= 8
    assert SECRET not in str(error) and SECRET not in repr(vars(error))
    assert 'input' not in vars(error) and 'loc' not in vars(error) and 'ctx' not in vars(error)
    assert error.__context__ is None and error.__cause__ is None
    return error


@pytest.mark.parametrize('rule', ['operation_field', 'operation_address', 'operation_value',
    'question_field', 'question_rows_limit', 'question_address', 'link_limit',
    'link_field', 'link_address', 'link_duplicate'])
def test_each_manual_rejection_has_specific_static_rule_and_no_candidate_metadata(case, rule):
    config, names = case
    candidate = response(ops=[op('set', 1, names['asset'], 'camera')])
    if rule == 'operation_field':
        candidate['ops'][0]['slot'] = SECRET
    elif rule == 'operation_address':
        candidate['ops'][0]['item_id'] = None
    elif rule == 'operation_value':
        candidate['ops'][0]['value'] = SECRET
    else:
        candidate = question(names)
        scope = candidate['clarification']
        if rule == 'question_field':
            scope['slot'] = SECRET
        elif rule == 'question_rows_limit':
            config['demo']['collection_max_items'] = 1
            scope.update(kind='item_reference', item_ids=[1, 2])
        elif rule == 'question_address':
            scope['slot'] = names['date']
        else:
            scope['linked_addresses'] = [address(1, names['quantity'])]
            if rule == 'link_limit':
                config['demo']['collection_max_items'] = 1
                scope['linked_addresses'] += [address(None, names['date']), address(2, names['quantity'])]
            elif rule == 'link_field':
                scope['linked_addresses'][0]['slot'] = SECRET
            elif rule == 'link_address':
                scope['linked_addresses'][0]['item_id'] = None
            else:
                scope['linked_addresses'].append(deepcopy(scope['linked_addresses'][0]))
    reject(candidate, config, 'collection_' + rule)


@pytest.mark.parametrize('rule', ['operation_shape', 'question_shape', 'contradictory_flags',
    'unresolved_mutation', 'intent_question', 'resolution_shape', 'proposal_shape',
    'discard_shape', 'request_discard'])
def test_custom_coherence_rules_remain_rejections_without_interpreting_candidate(case, rule):
    config, names = case
    candidate = response()
    if rule == 'operation_shape':
        candidate['ops'] = [op('clear', 1, names['asset'], SECRET)]
    elif rule == 'question_shape':
        candidate = question(names, item_ids=[1, 1])
    elif rule == 'contradictory_flags':
        candidate.update(is_affirmation=True, is_negation=True)
    elif rule == 'unresolved_mutation':
        candidate.update(intent='help', ops=[op('set', 1, names['asset'], 'camera')])
    elif rule == 'intent_question':
        candidate.update(intent='ambiguous')
    elif rule == 'resolution_shape':
        candidate['resolves_clarification'] = 1
    elif rule == 'proposal_shape':
        candidate['proposed_ops'] = [op('set', 1, names['asset'], SECRET)]
    elif rule == 'discard_shape':
        candidate.update(discard_clarification=1, ops=[op('set', 1, names['asset'], 'camera')])
    else:
        candidate.update(discard_request=SECRET, ops=[op('set', 1, names['asset'], 'camera')])
    reject(candidate, config, 'collection_' + rule)


@pytest.mark.parametrize('category', ['json', 'required', 'extra', 'type', 'constraint'])
def test_builtin_shape_errors_use_coarse_whitelisted_categories(case, category):
    config, _ = case
    candidate = response()
    if category == 'json':
        candidate = SECRET
    elif category == 'required':
        candidate.pop('intent')
    elif category == 'extra':
        candidate['ops'] = [dict(op='create', item_id=1, slot=None, value=None, **{SECRET: SECRET})]
    elif category == 'type':
        candidate['ops'] = SECRET
    else:
        candidate['confidence'] = 99
    reject(candidate, config, 'collection_shape_' + category)


@pytest.mark.parametrize('renamed', [False, True])
def test_valid_cross_address_payloads_keep_exact_acceptance_and_values(monkeypatch, renamed):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    config, names = collection_config(renamed=renamed)
    value = question(names, linked_addresses=[address(2, names['quantity']), address(None, names['date'])])
    value['proposed_ops'] = [op('create', 1), op('create', 2)]
    parsed = parse_collection_response(json.dumps(value), config)
    assert parsed.model_dump() == value
    direct = response(ops=row(names, 1) + [op('set', None, names['date'], '2026-09-15')])
    assert parse_collection_response(json.dumps(direct), config).model_dump() == direct


def test_unknown_custom_error_codes_and_input_locations_cannot_become_provenance():
    raw_error = ValidationError.from_exception_data('Synthetic validation', [
        dict(type=PydanticCustomError(SECRET, '{value}', {'value': SECRET}), loc=(SECRET,), input=SECRET),
        dict(type='extra_forbidden', loc=('ops', 0, SECRET), input=SECRET)])
    assert SECRET in str(raw_error), 'The fixture must contain actual unsafe validation metadata'
    safe = CollectionValidationError(validation_error_rules(raw_error))
    assert safe.validation_rules == ('collection_shape_extra', 'collection_shape_other')
    assert SECRET not in str(safe) and SECRET not in repr(vars(safe))
    assert safe.rules_truncated is False


def test_provenance_is_bounded_deduplicated_and_unknown_rules_collapse_to_static_other():
    safe = CollectionValidationError(list(VALIDATION_RULES) + [SECRET, SECRET, {'input': SECRET}])
    assert len(safe.validation_rules) == 8 and safe.rules_truncated is True
    assert set(safe.validation_rules) <= VALIDATION_RULES
    assert list(safe.validation_rules) == sorted(set(safe.validation_rules))
    assert SECRET not in str(safe) and SECRET not in repr(vars(safe))


def test_invalid_configuration_is_not_reclassified_as_rejected_model_output(case):
    config, _ = case
    config['demo']['transaction_schema']['root_slots'] = []
    with pytest.raises(ValueError) as error:
        parse_collection_response(json.dumps(response(intent='help')), config)
    assert not isinstance(error.value, CollectionValidationError)


def provider_payload(candidate):
    return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
        'message': {'content': json.dumps(candidate)}}]})


@pytest.mark.parametrize('recover', [False, True])
def test_router_retries_keep_state_and_exclude_untrusted_fields_from_all_diagnostics(case, tmp_path, recover):
    config, names = case
    config['router']['retries'] = 1
    task = ConfiguredCollectionState(config)
    task.apply(row(names, 1))
    task.consume(question(names, linked_addresses=[address(1, names['quantity'])]))
    original, context = deepcopy(task.__dict__), task.router_context()
    bad = response(ops=[dict(op='set', item_id=1, slot=SECRET, value=SECRET, **{SECRET: SECRET})])
    requests, reserves = [], []

    async def run():
        def http(request):
            requests.append(json.loads(request.content))
            return provider_payload(response(intent='help') if recover and len(requests) == 2 else bad)
        async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
            router = Router(config, tmp_path, api_key='offline-fixture', client=client)
            router.limiter.reserve = lambda: reserves.append(True)
            if recover:
                assert (await router.route('please explain the choices', context)).intent == 'help'
            else:
                with pytest.raises(RouterOutputError) as error:
                    await router.route('please explain the choices', context)
                assert SECRET not in str(error.value)
            assert len(requests) == len(reserves) == len(router.calls) == 2
            assert task.__dict__ == original and context == task.router_context()
            for item in router.calls:
                if not item['ok']:
                    assert item['validation_stage'] == 'response_schema'
                    assert item['validation_rules'] == ['collection_shape_extra']
                    assert 'validation_errors' not in item and 'response_shape' not in item
            assert SECRET not in json.dumps(router.calls)
            assert SECRET not in json.dumps(requests)
    asyncio.run(run())


def test_pipeline_repair_and_saved_report_cannot_echo_rejected_collection_payload(case, tmp_path):
    config, names = case
    config['router']['retries'] = 0
    events = []
    bad = response(ops=[op('set', 1, names['asset'], SECRET)])

    async def run():
        class ASR:
            async def transcribe(self, pcm):
                return Transcript('a nonexact new request', .99, 'offline-fixture')
        class Bank:
            async def render(self, action, values):
                return RenderedAudio(wav_bytes(bytes(1024), config['runtime']), 512, 512, ['fixture'], 'Offline response')
        async def emit(event):
            events.append(deepcopy(event))
            if event['type'] == 'audio_end':
                await voice.playback_finished(event['playback_id'])
        async def audio(data):
            pass
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: provider_payload(bad))) as client:
            router = Router(config, tmp_path, api_key='offline-fixture', client=client)
            router.limiter.reserve = lambda: None
            voice = VoiceSession(config, tmp_path, ASR(), router, Bank(), None, emit, audio)
            voice.task.apply(row(names, 1))
            original = deepcopy(voice.task.values)
            try:
                await voice.start()
                await voice.playback_task
                await voice.queue.put(Segment(bytes([1, 0]) * 512, 0, 320, 1728, 320))
                await asyncio.wait_for(voice.queue.join(), 2)
                await voice.playback_task
                assert voice.task.values == original and not voice.task.confirmed
                assert any(event['type'] == 'warning' for event in events)
                assert router.calls[0]['validation_rules'] == ['collection_operation_value']
                assert SECRET not in json.dumps(events) and SECRET not in json.dumps(voice.turns)
            finally:
                await voice.close()
            reports = list(tmp_path.rglob('demo_session_*.json'))
            assert len(reports) == 1 and SECRET not in reports[0].read_text(encoding='utf-8')
    asyncio.run(run())
