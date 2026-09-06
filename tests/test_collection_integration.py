"""Temporary collection profiles across discovery, rendering, sessions and recovery."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import re

import httpx
import pytest
from jsonschema import Draft202012Validator

from collection_fixtures import write_collection_fixture
from engine.collection_task import ConfiguredCollectionState
from engine.config import load_config
import engine.demo as demo
from engine.endpointing import Segment
from engine.pipeline import VoiceSession
from engine.recovery import RecoveryStore, restore_values
from engine.responder import RenderedAudio
from engine.router import Router, RouterOutputError
import engine.server as server
from engine.state import Action
from engine.stt import Transcript, wav_bytes
from engine.task_factory import make_task


@pytest.fixture(params=[False, True], ids=['equipment', 'renamed'])
def case(request, monkeypatch, tmp_path):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    monkeypatch.setenv('GROQ_API_KEY', 'offline-fixture')
    monkeypatch.setenv('MOULSOT_ENDPOINT', 'http://127.0.0.1:8012/transcribe')
    monkeypatch.setenv('MOULSOT_PROTOCOL', 'json')
    files = write_collection_fixture(tmp_path, renamed=request.param)
    monkeypatch.setattr(demo, 'ROOT', tmp_path)
    monkeypatch.setattr(server, 'ROOT', tmp_path)
    # index renders the real frontend template, with no provider construction.
    web = tmp_path / 'web'
    web.mkdir()
    web.joinpath('index.html').write_bytes((Path(__file__).resolve().parents[1] / 'web/index.html').read_bytes())
    return files


def config_for(case):
    return demo.demo_config(load_config(case['config_path']))


def op(kind, item_id=None, slot=None, value=None):
    return dict(op=kind, item_id=item_id, slot=slot, value=value)


def row(names, item_id, asset='camera', quantity=1):
    return [op('create', item_id), op('set', item_id, names['asset'], asset),
            op('set', item_id, names['quantity'], quantity)]


def response(**changes):
    result = dict(intent='task', ops=[], confidence=.99, unclear=False,
        is_affirmation=False, is_negation=False, clarification=None, proposed_ops=[],
        resolves_clarification=None, discard_clarification=None, discard_request=None)
    result.update(changes)
    if result['clarification'] is not None:
        result['clarification'] = {**result['clarification']}
        result['clarification'].setdefault('linked_addresses', None)
    return result


def test_full_profile_discovery_validation_and_public_collection_metadata(case):
    config, names = config_for(case), case['names']
    assert demo.demo_supported(load_config(case['config_path']))
    assert {'id': names['domain'], 'title': case['base']['display_name']} in server.available_domains()
    task = make_task(config)
    assert isinstance(task, ConfiguredCollectionState)
    page = asyncio.run(server.index(domain=names['domain'], mode='demo'))
    assert page.status_code == 200
    public = json.loads(re.search(r'<script id="capture-config" type="application/json">(.*?)</script>',
        page.body.decode(), re.S).group(1))
    assert public['demo_supported'] and public['demo_state_kind'] == 'configured_collection_scoped'
    assert public['demo_collection'] == dict(name=names['collection'], label='Reservation',
        root_slots=[names['date']], item_slots=[names['asset'], names['quantity']], min_items=1, max_items=10)
    assert public['demo_issues'] == []


@pytest.mark.parametrize('invalid', ['partition', 'label', 'format'])
def test_bad_merged_profile_fails_before_any_task_or_provider(case, invalid):
    profile, names = deepcopy(case['profile']), case['names']
    if invalid == 'partition':
        profile['transaction_schema']['root_slots'].append(names['asset'])
    elif invalid == 'label':
        profile['labels'].pop(names['quantity'])
    else:
        profile['readback_formats'][names['asset']] = 'day_month_year'
    case['profile_path'].write_text(json.dumps(profile), encoding='utf-8')
    with pytest.raises(ValueError, match='(?i)field|label|format|slot'):
        config_for(case)


def test_factory_pipeline_and_restore_use_same_generic_adapter_with_fresh_ids(case):
    config, names = config_for(case), case['names']
    task = make_task(config)
    task.apply(row(names, 1) + row(names, 2, 'tripod', 2) + [op('delete', 1)] +
        row(names, 3, 'camera', 3) + [op('set', None, names['date'], '2026-09-15')])
    async def emit(event):
        pass
    voice = VoiceSession(config, case['root'], None, None, None, None, emit, emit)
    restored = restore_values(config, task.values)
    assert type(task) is type(voice.task) is type(restored) is ConfiguredCollectionState
    assert [item['id'] for item in restored.values[names['collection']]] == [1, 2]
    assert restored.next_item_id == 3 and restored.readback_version is None and not restored.confirmed
    text = demo.reply_text(Action('readback'), restored.values, config['demo'])
    assert 'Reservation 1.' in text and 'Reservation 2.' in text
    assert 'Tripod' in text and 'Camera' in text and '15 / 09 / 2026' in text
    assert 'pizza' not in text.lower() and 'doctor' not in text.lower()


@pytest.mark.parametrize('change', ['label', 'limit', 'field_partition'])
def test_recovery_rejects_changed_collection_meaning_without_consuming_offer(case, change):
    config, names = config_for(case), case['names']
    task = make_task(config)
    task.apply(row(names, 1) + [op('set', None, names['date'], '2026-09-15')])
    registry = RecoveryStore(clock=lambda: 100)
    offer = registry.offer(config, task, 'offline-source')
    changed = deepcopy(config)
    if change == 'label':
        changed['demo']['collection_label'] = 'Loan'
    elif change == 'limit':
        changed['demo']['collection_max_items'] = 9
    else:
        changed['demo']['transaction_schema']['item_slots'].remove(names['asset'])
        changed['demo']['transaction_schema']['root_slots'].append(names['asset'])
    with pytest.raises(ValueError):
        registry.restore(offer['token'], changed)
    restored = registry.restore(offer['token'], config)
    assert restored.task.values == task.values and not restored.task.confirmed


class Harness:
    """Actual pipeline and rendering; explicit offline router dictionaries."""
    def __init__(self, config, root):
        self.config, self.text = config, ''
        self.candidates, self.contexts, self.events = [], [], []
        self.auto_ack = True
        self.ended = asyncio.Queue()
        self.voice = VoiceSession(config, root, self, self, self, None, self.emit, self.emit_audio)
    async def route(self, text, context):
        self.contexts.append(deepcopy(context))
        assert self.candidates, 'Unexpected router turn'
        return self.candidates.pop(0)
    async def transcribe(self, pcm):
        return Transcript(self.text, .99, 'offline-fixture')
    async def render(self, action, values):
        return RenderedAudio(wav_bytes(bytes(1024), self.config['runtime']), 512, 512,
            ['fixture'], demo.reply_text(action, values, self.config['demo']))
    async def emit(self, event):
        self.events.append(deepcopy(event))
        if event['type'] == 'audio_end':
            if self.auto_ack:
                await self.voice.playback_finished(event['playback_id'])
            else:
                self.ended.put_nowait(event['playback_id'])
    async def emit_audio(self, data):
        pass
    async def start(self):
        await self.voice.start()
        await asyncio.wait_for(self.voice.playback_task, 2)
    async def speak(self, text, value):
        self.text = text
        self.candidates.append(value)
        await self.voice.queue.put(Segment(bytes([len(self.contexts) + 1, 0]) * 512, 0, 320, 1728, 320))
        await asyncio.wait_for(self.voice.queue.join(), 2)
        if self.auto_ack:
            await asyncio.wait_for(self.voice.playback_task, 2)


def test_full_session_keeps_separate_rows_and_confirms_only_after_fresh_readback(case):
    async def run():
        config, names = config_for(case), case['names']
        harness = Harness(config, case['root'])
        voice = harness.voice
        try:
            await harness.start()
            await harness.speak('camera and tripod', response(ops=row(names, 1) + row(names, 2, 'tripod', 2) +
                [op('set', None, names['date'], '2026-09-15')]))
            first = deepcopy(voice.task.values[names['collection']][0])
            await harness.speak('change second quantity', response(ops=[op('set', 2, names['quantity'], 4)]))
            assert voice.task.values[names['collection']][0] == first
            assert voice.task.values[names['collection']][1][names['quantity']] == 4
            old_ack = next(event['playback_id'] for event in reversed(harness.events) if event['type'] == 'audio_end')
            harness.auto_ack = False
            await harness.speak('delete first then add camera', response(ops=[op('delete', 1)] + row(names, 3, 'camera', 3)))
            fresh = await asyncio.wait_for(harness.ended.get(), 2)
            assert [item['id'] for item in voice.task.values[names['collection']]] == [2, 3]
            assert voice.task.readback_version is None and not voice.task.confirmed
            await voice.playback_finished(old_ack)
            await asyncio.sleep(0)
            assert voice.task.readback_version is None
            await voice.playback_finished(fresh)
            await asyncio.wait_for(voice.playback_task, 2)
            assert voice.task.readback_version == voice.task.version
            harness.auto_ack = True
            await harness.speak('yes', response(is_affirmation=True))
            assert voice.task.confirmed and voice.done.is_set()
            assert not any(event['type'] == 'error' for event in harness.events)
        finally:
            await voice.close()
    asyncio.run(run())


def test_draft_question_uses_preview_ordinal_but_trace_and_recovery_remain_committed(case):
    async def run():
        config, names = config_for(case), case['names']
        harness = Harness(config, case['root'])
        voice = harness.voice
        voice.task.apply(row(names, 1) + [op('set', None, names['date'], '2026-09-15')])
        original = deepcopy(voice.task.values)
        try:
            await harness.start()
            text = 'add camera and another unsupported equipment choice'
            proposal = row(names, 2) + [op('create', 3), op('set', 3, names['quantity'], 2)]
            await harness.speak(text, response(intent='out_of_scope', proposed_ops=proposal,
                clarification=dict(kind='unsupported_value', slot=names['asset'], item_ids=[3])))
            assert voice.task.values == original and voice.task.next_item_id == 2
            assert voice.pending_request['text'] == text
            spoken = [event['text'] for event in harness.events if event['type'] == 'assistant_text'][-1]
            assert spoken.startswith('Reservation 3.'), spoken
            assert voice.turns[-1]['state'] == original
            assert voice.task.pending_proposal['next_item_id'] == 4
            registry = RecoveryStore(clock=lambda: 100)
            offer = registry.offer(config, voice.task, voice.session_id, pending_request=voice.pending_request)
            assert offer['slots'] == original and offer['discarded_pending']
            restored = registry.restore(offer['token'], config).task
            assert restored.values == original and restored.pending_proposal is None
            question_id = voice.task.pending_clarification['id']
            await harness.speak('tripod', response(ops=[op('set', 3, names['asset'], 'tripod')],
                resolves_clarification=question_id))
            assert voice.pending_request is None and voice.task.next_item_id == 4
            assert [item['id'] for item in voice.task.values[names['collection']]] == [1, 2, 3]
            assert harness.contexts[-1]['pending_request']['text'] == text
        finally:
            await voice.close()
    asyncio.run(run())


@pytest.mark.parametrize('destructive', ['clear', 'delete'])
def test_answer_followed_by_destructive_operation_cannot_resolve_value_question(case, destructive):
    config, names = config_for(case), case['names']
    task = make_task(config)
    task.consume(response(intent='out_of_scope', proposed_ops=[op('create', 1),
        op('set', 1, names['quantity'], 2)], clarification=dict(kind='unsupported_value', slot=names['asset'], item_ids=[1])))
    before = deepcopy(task.__dict__)
    extra = op('delete', 1) if destructive == 'delete' else op('clear', 1, names['asset'])
    with pytest.raises(ValueError):
        task.consume(response(ops=[op('set', 1, names['asset'], 'camera'), extra],
            resolves_clarification=task.pending_clarification['id']))
    assert task.__dict__ == before


def test_collection_wire_schema_and_parser_enforce_configured_field_addressing(case):
    from engine.collection_routing import collection_response_format, parse_collection_response, collection_prompt
    config, names = config_for(case), case['names']
    schema = collection_response_format(config)['json_schema']['schema']
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    valid = response(ops=row(names, 1) + [op('set', None, names['date'], '2026-09-15')])
    validator.validate(valid)
    parsed = parse_collection_response(json.dumps(valid), config)
    task = make_task(config)
    assert task.consume(parsed.model_dump()).kind == 'readback'
    # Groq accepts a structural object schema. Address/type relationships remain
    # authoritative local parser checks rather than overlapping object unions.
    for invalid in [response(ops=[op('set', None, names['asset'], 'camera')]),
        response(ops=[op('set', 1, names['date'], '2026-09-15')]),
        response(intent='ambiguous', clarification=dict(kind='ambiguous_value', slot=names['asset'], item_ids=[])),
        response(intent='ambiguous', clarification=dict(kind='ambiguous_value', slot=names['date'], item_ids=[1])),
        response(ops=[op('clear', 1, names['asset'], 'camera')])]:
        validator.validate(invalid)
        with pytest.raises(ValueError):
            parse_collection_response(json.dumps(invalid), config)
    for invalid in [response(ops=[op('set', True, names['asset'], 'camera')]),
        response(intent='ambiguous', clarification=dict(kind='ambiguous_value', slot=None, item_ids=[])),
        response(intent='out_of_scope', clarification=dict(kind='unsupported_value', slot=None, item_ids=[]))]:
        assert not validator.is_valid(invalid)
        with pytest.raises(ValueError):
            parse_collection_response(json.dumps(invalid), config)
    prompt = collection_prompt(config, task.router_context(), 'change second reservation')
    assert names['collection'] in prompt and names['asset'] in prompt
    assert all(word not in prompt.lower() for word in ('pizza', 'doctor', 'clinic'))


async def route_http(config, root, transcript, context, candidates, observed):
    """Exercise production request shaping, parsing and guards with HTTP fixtures."""
    before = deepcopy(context)
    def reply(request):
        observed.setdefault('requests', []).append(json.loads(request.content))
        assert candidates, 'Unexpected extra HTTP router attempt'
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps(candidates.pop(0))}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        router = Router(config, root, api_key='offline-fixture', client=client)
        router.limiter.reserve = lambda: None
        try:
            return await router.route(transcript, context)
        finally:
            observed['calls'] = deepcopy(router.calls)
            assert context == before


def test_real_router_sends_generic_collection_schema_and_retries_bad_addressing(case):
    config, names = config_for(case), case['names']
    config['router']['retries'] = 1
    task, observed = make_task(config), {}
    bad = response(ops=[op('set', None, names['asset'], 'camera')])
    good = response(ops=row(names, 1) + [op('set', None, names['date'], '2026-09-15')])
    parsed = asyncio.run(route_http(config, case['root'], 'camera quantity one return September fifteen',
        task.router_context(), [bad, good], observed))
    assert len(observed['requests']) == 2
    assert observed['calls'][0]['validation_stage'] == 'response_schema'
    assert observed['calls'][1]['ok']
    schema = observed['requests'][0]['response_format']['json_schema']['schema']
    validator = Draft202012Validator(schema)
    validator.validate(good)
    # Wire-valid structural output still requires local addressing validation.
    validator.validate(bad)
    messages = observed['requests'][0]['messages']
    assert names['collection'] in messages[0]['content']
    assert messages[-1] == {'role': 'user', 'content': 'camera quantity one return September fifteen'}
    assert all(word not in messages[0]['content'].lower() for word in ('pizza', 'doctor', 'clinic'))
    assert task.values == {names['collection']: []}
    assert task.consume(parsed.model_dump()).kind == 'readback'


@pytest.mark.parametrize('identity', ['missing', 'stale'])
def test_real_router_requires_current_draft_question_id_and_preserves_preview_history(case, identity):
    config, names = config_for(case), case['names']
    config['router']['retries'] = 1
    task, observed = make_task(config), {}
    task.consume(response(intent='out_of_scope', proposed_ops=[op('create', 1),
        op('set', 1, names['quantity'], 2)], clarification=dict(kind='unsupported_value', slot=names['asset'], item_ids=[1])))
    pending_id = task.pending_clarification['id']
    before = deepcopy(task.__dict__)
    context = task.router_context()
    context.update(pending_request={'id': 'request_1', 'text': 'original equipment request', 'truncated': False},
        last_assistant_text='Reservation 1. Which equipment?')
    candidate = response(ops=[op('set', 1, names['asset'], 'tripod')],
        resolves_clarification=None if identity == 'missing' else pending_id + 1)
    valid = response(ops=[op('set', 1, names['asset'], 'tripod')], resolves_clarification=pending_id)
    parsed = asyncio.run(route_http(config, case['root'], 'tripod', context, [candidate, valid], observed))
    assert task.__dict__ == before
    assert observed['calls'][0]['validation_stage'] == 'clarification_identity'
    assert len(observed['calls']) == 2 and observed['calls'][1]['ok']
    messages = observed['requests'][-1]['messages']
    assert {'role': 'user', 'content': 'original equipment request'} in messages
    assert {'role': 'assistant', 'content': 'Reservation 1. Which equipment?'} in messages
    assert 'committed_state' in messages[0]['content'] and 'next_item_id' in messages[0]['content']
    task.consume(parsed.model_dump())
    assert task.values[names['collection']] == [{'id': 1, names['quantity']: 2, names['asset']: 'tripod'}]
    assert task.pending_proposal is None


def test_real_router_stale_retained_request_discard_exhausts_without_mutation(case):
    config, names = config_for(case), case['names']
    config['router']['retries'] = 1
    task, observed = make_task(config), {}
    task.apply(row(names, 1))
    context = task.router_context()
    context.update(pending_request={'id': 'request_7', 'text': 'original return date request',
        'clarification_resolved': True}, last_assistant_text='Which return date?')
    candidate = response(discard_request='request_6')
    before = deepcopy(task.__dict__)
    with pytest.raises(RouterOutputError):
        asyncio.run(route_http(config, case['root'], 'forget previous request', context,
            [candidate, candidate], observed))
    assert task.__dict__ == before
    assert len(observed['calls']) == 2
    assert all(call['validation_stage'] == 'retained_request_discard' and not call['ok'] for call in observed['calls'])
    valid = asyncio.run(route_http(config, case['root'], 'forget previous request', context,
        [response(discard_request='request_7')], {}))
    task.consume(valid.model_dump(), retained_request=context['pending_request'])
    assert task.values == before['values'] and task.version == before['version'] + 1


def test_profile_required_row_promotion_is_validated_after_effective_settings(case):
    names = case['names']
    base, profile = deepcopy(case['base']), deepcopy(case['profile'])
    for slot in base['slots']:
        if slot['id'] in (names['asset'], names['quantity']):
            slot['required'] = False
    profile['required_slots'] = [names['asset'], names['date']]
    case['config_path'].write_text(json.dumps(base), encoding='utf-8')
    case['profile_path'].write_text(json.dumps(profile), encoding='utf-8')
    original = load_config(case['config_path'])
    config = demo.demo_config(original)
    flags = {slot['id']: slot['required'] for slot in config['slots']}
    assert flags[names['asset']] is True and flags[names['quantity']] is False
    assert all(not slot['required'] for slot in original['slots'] if slot['id'] != names['date'])
    task = make_task(config)
    assert task.next_action().slot == names['asset'] and task.next_action().item_id == 1
    task.apply([op('create', 1), op('set', 1, names['asset'], 'camera'),
                op('set', None, names['date'], '2026-09-15')])
    assert task.ready and task.next_action().kind == 'readback'


def test_profile_without_any_effectively_required_row_field_still_rejects(case):
    names = case['names']
    base, profile = deepcopy(case['base']), deepcopy(case['profile'])
    for slot in base['slots']:
        if slot['id'] in (names['asset'], names['quantity']):
            slot['required'] = False
    profile['required_slots'] = [names['date']]
    case['config_path'].write_text(json.dumps(base), encoding='utf-8')
    case['profile_path'].write_text(json.dumps(profile), encoding='utf-8')
    with pytest.raises(ValueError, match='(?i)required row'):
        config_for(case)


def test_required_promotion_does_not_hide_a_malformed_override_flag(case):
    names = case['names']
    profile = deepcopy(case['profile'])
    profile['slot_overrides'][names['asset']] = {'required': 'false'}
    assert names['asset'] in profile['required_slots']
    case['profile_path'].write_text(json.dumps(profile), encoding='utf-8')
    with pytest.raises(ValueError, match='(?i)required|boolean'):
        config_for(case)


def test_wire_object_unions_have_no_overlapping_operation_or_kind_discriminators(case):
    from engine.collection_routing import collection_response_format
    schema = collection_response_format(config_for(case))['json_schema']['schema']
    seen_discriminators = set()
    def inspect(node):
        if isinstance(node, dict):
            if node.get('type') == 'object':
                assert node.get('additionalProperties') is False
                assert set(node['required']) == set(node['properties'])
            for keyword in ('anyOf', 'oneOf'):
                branches = [branch for branch in node.get(keyword, []) if branch.get('type') == 'object']
                if len(branches) > 1:
                    for key in ('op', 'kind'):
                        if all('enum' in branch.get('properties', {}).get(key, {}) for branch in branches):
                            choices = [choice for branch in branches for choice in branch['properties'][key]['enum']]
                            assert len(choices) == len(set(choices)), (key, choices)
                            seen_discriminators.add(key)
            for value in node.values():
                inspect(value)
        elif isinstance(node, list):
            for value in node:
                inspect(value)
    inspect(schema)
    assert 'kind' in seen_discriminators
