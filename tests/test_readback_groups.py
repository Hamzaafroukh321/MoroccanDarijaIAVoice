"""Opt-in readback cache grouping with fake speech and no provider requests."""
import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from engine.config import ROOT, load_config
import engine.demo as demo
from engine.pipeline import VoiceSession
from engine.responder import AudioBankError
from engine.state import Action
from engine.stt import wav_bytes


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'elevenlabs')
    monkeypatch.setenv('ELEVENLABS_API_KEY', 'offline-fixture')
    result = demo.demo_config(load_config(ROOT / 'configs/clinic.json'))
    result['demo'].update(tts_readback_mode='grouped', tts_readback_group_chars=96, cache_dir='cache')
    return result


def values():
    return {'doctor': 'doctor_a', 'date': '2026-09-15', 'time': '10:30'}


def cache(config, root, text, pcm):
    path = demo.speech_cache_path(config['demo'], root, text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav_bytes(pcm, config['runtime']))
    return path


def test_flat_pieces_preserve_exact_configured_text_and_optional_fields(config):
    settings = deepcopy(config['demo'])
    settings.update(state_kind='flat_scoped', readback_prefix='Preferences:', readback_question='Correct?',
        labels={'items': 'Counters', 'date': 'Day', 'time': 'Time', 'note': 'Note'},
        values={'a': 'Counter A', 'b': 'Counter B'}, readback_order=['items', 'date', 'time', 'note'])
    state = {'items': ['a', 'b'], 'date': '2026-09-15', 'time': '11:30'}
    expected = 'Preferences: Counters: Counter A، Counter B. Day: 15 / 09 / 2026. Time: 11 : 30. Correct?'
    assert ' '.join(demo.readback_parts(state, settings)) == expected
    assert demo.reply_text(Action('readback'), state, settings) == expected
    assert ' '.join(demo.readback_groups(state, settings, 48)) == expected


def test_pizza_pieces_preserve_item_order_plain_toppings_and_drinks(monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'elevenlabs')
    config = demo.demo_config(load_config(ROOT / 'configs/pizza.json'))
    settings = config['demo']
    state = {'items': [{'id': 2, 'quantity': 1, 'size': 'small', 'toppings': []},
                       {'id': 7, 'quantity': 2, 'size': 'large', 'toppings': ['cheese']}], 'drink': ['water']}
    expected = [settings['readback_prefix']]
    for index, item in enumerate(state['items'], 1):
        expected.append(f"{settings['item_label']} {index}.")
        for field in ('quantity', 'size', 'toppings'):
            raw = item[field]
            rendered = settings['plain_toppings'] if raw == [] else settings['values'].get(
                str(raw[0] if isinstance(raw, list) else raw), str(raw))
            expected.append(settings['labels'][field] + ': ' + rendered + '.')
    expected += [settings['labels']['drink'] + ': ' + settings['values']['water'] + '.', settings['readback_question']]
    text = ' '.join(expected)
    assert ' '.join(demo.readback_parts(state, settings)) == text
    assert demo.reply_text(Action('readback'), state, settings) == text
    assert ' '.join(demo.readback_groups(state, settings, 96)) == text


def test_long_individual_piece_uses_existing_length_chunks(config):
    settings = deepcopy(config['demo'])
    settings.update(readback_prefix='start', readback_question='end', labels={'note': 'Note'},
                    readback_order=['note'], readback_formats={})
    state = {'note': ' '.join(['longword'] * 20)}
    groups = list(demo.readback_groups(state, settings, 32))
    assert all(len(part) <= 32 for part in groups)
    assert ' '.join(groups) == demo.reply_text(Action('readback'), state, settings)


@pytest.mark.parametrize('legacy_maximum', [1000, 55])
def test_existing_complete_legacy_cache_wins_over_grouped_plan(config, tmp_path, legacy_maximum):
    config['demo']['tts_max_chars'] = legacy_maximum
    config['demo']['tts_readback_group_chars'] = min(48, legacy_maximum)
    text = demo.reply_text(Action('readback'), values(), config['demo'])
    parts = list(demo.chunks(text, legacy_maximum))
    expected = []
    for index, part in enumerate(parts, 1):
        pcm = bytes([index, 0]) * 512
        expected.append(pcm)
        cache(config, tmp_path, part, pcm)
    def forbidden(request):
        pytest.fail('Existing complete legacy cache must avoid speech generation')
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
            voice = demo.DemoVoice(config, tmp_path, client=client)
            rendered = await voice.render(Action('readback'), values())
            assert demo.pcm16(rendered.wav, config['runtime']) == b''.join(expected)
            assert voice.calls[-1]['render_mode'] == 'whole'
            assert voice.calls[-1]['cache_hits'] == len(parts)
            await voice.close()
    asyncio.run(run())


def test_changed_time_reuses_unchanged_groups_without_publishing_assembled_whole(config, tmp_path):
    requested = []
    def handle(request):
        requested.append(json.loads(request.content)['text'])
        return httpx.Response(200, content=bytes([len(requested), 0]) * 512)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            voice = demo.DemoVoice(config, tmp_path, client=client)
            initial = values()
            first = await voice.render(Action('readback'), initial)
            initial_count = len(requested)
            assert initial_count > 1 and voice.calls[-1]['render_mode'] == 'grouped'
            changed = {**initial, 'time': '11:00'}
            second = await voice.render(Action('readback'), changed)
            assert len(requested) == initial_count + 1
            assert voice.calls[-1]['cache_misses'] == 1
            assert voice.calls[-1]['cache_hits'] == initial_count - 1
            third = await voice.render(Action('readback'), changed)
            assert third.wav == second.wav and len(requested) == initial_count + 1
            assert voice.calls[-1]['cache_misses'] == 0
            assert first.text == demo.reply_text(Action('readback'), initial, config['demo'])
            assert second.text == demo.reply_text(Action('readback'), changed, config['demo'])
            for rendered in (first, second):
                assert not demo.speech_cache_path(config['demo'], tmp_path, rendered.text).exists()
            await voice.close()
    asyncio.run(run())


def test_default_whole_mode_and_non_readback_actions_keep_legacy_generation(config, tmp_path):
    config['demo'].pop('tts_readback_mode')
    seen = []
    def handle(request):
        seen.append(json.loads(request.content)['text'])
        return httpx.Response(200, content=bytes(1024))
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            voice = demo.DemoVoice(config, tmp_path, client=client)
            rendered = await voice.render(Action('readback'), values())
            assert seen == [rendered.text]
            assert voice.calls[-1]['render_mode'] == 'whole'
            config['demo']['tts_readback_mode'] = 'grouped'
            await voice.render(Action('greeting'), {})
            assert voice.calls[-1]['render_mode'] == 'whole'
            assert len(seen) == 2
            await voice.close()
    asyncio.run(run())


def test_excessive_group_count_falls_back_to_legacy_plan(config, tmp_path):
    settings = config['demo']
    settings.update(tts_readback_group_chars=32, tts_max_chars=1000,
        readback_prefix='Start', readback_question='End', readback_order=['note'],
        readback_formats={}, labels={'note': 'Note'})
    state = {'note': ' '.join(['longword'] * 90)}
    assert len(list(demo.readback_groups(state, settings, 32))) > 16
    seen = []
    def handle(request):
        seen.append(json.loads(request.content)['text'])
        return httpx.Response(200, content=bytes(1024))
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            voice = demo.DemoVoice(config, tmp_path, client=client)
            result = await voice.render(Action('readback'), state)
            assert seen == [result.text]
            assert voice.calls[-1]['render_mode'] == 'whole'
            await voice.close()
    asyncio.run(run())


@pytest.mark.parametrize('field,value', [('tts_readback_mode', 'stream'), ('tts_readback_mode', None),
    ('tts_readback_group_chars', True), ('tts_readback_group_chars', 31),
    ('tts_readback_group_chars', 96.0), ('tts_readback_group_chars', 100000)])
def test_invalid_group_settings_fail_preflight(config, field, value):
    config['demo'][field] = value
    with pytest.raises(demo.DemoConfigError, match='tts_readback'):
        demo.validate_demo_config(config)


def test_omitted_group_limit_clamps_to_provider_maximum(config, tmp_path):
    config['demo'].pop('tts_readback_group_chars')
    config['demo']['tts_max_chars'] = 55
    demo.validate_demo_config(config)
    seen = []
    def handle(request):
        seen.append(json.loads(request.content)['text'])
        return httpx.Response(200, content=bytes(1024))
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            voice = demo.DemoVoice(config, tmp_path, client=client)
            result = await voice.render(Action('readback'), values())
            assert all(len(part) <= 55 for part in seen)
            assert ' '.join(seen) == result.text
            assert voice.calls[-1]['render_mode'] == 'grouped'
            assert voice.calls[-1]['parts'] == len(seen)
            await voice.close()
    asyncio.run(run())


def test_valid_token_longer_than_group_limit_falls_back_to_provider_length_plan(config, tmp_path):
    config['demo']['tts_max_chars'] = 150
    config['demo']['values']['doctor_a'] = 'x' * 100
    demo.validate_demo_config(config)
    text = demo.reply_text(Action('readback'), values(), config['demo'])
    expected = list(demo.chunks(text, 150))
    seen = []
    def handle(request):
        seen.append(json.loads(request.content)['text'])
        return httpx.Response(200, content=bytes(1024))
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            voice = demo.DemoVoice(config, tmp_path, client=client)
            result = await voice.render(Action('readback'), values())
            assert result.text == text and seen == expected
            assert voice.calls[-1]['render_mode'] == 'whole'
            assert voice.calls[-1]['parts'] == len(expected)
            await voice.close()
    asyncio.run(run())


@pytest.mark.parametrize('field', ['tts_voice', 'tts_model', 'tts_url', 'tts_cache_revision'])
def test_group_cache_isolated_when_synthesis_identity_changes(config, tmp_path, field):
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(200, content=bytes(1024))
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            voice = demo.DemoVoice(config, tmp_path, client=client)
            await voice.render(Action('readback'), values())
            initial = len(requests)
            assert initial > 1
            config['demo'][field] = (str(config['demo'].get(field, '')) + '-changed')
            await voice.render(Action('readback'), values())
            assert len(requests) == initial * 2
            assert voice.calls[-1]['cache_hits'] == 0
            assert voice.calls[-1]['cache_misses'] == initial
            await voice.close()
    asyncio.run(run())


@pytest.mark.parametrize('outcome', ['complete', 'failure', 'cancel'])
def test_all_groups_finish_before_pipeline_audio_and_confirmation(config, tmp_path, outcome):
    async def run():
        entered, release, audio_end = asyncio.Event(), asyncio.Event(), asyncio.Event()
        requests, events, binary = [], [], []
        async def handle(request):
            requests.append(json.loads(request.content)['text'])
            if len(requests) == 2:
                entered.set()
                await release.wait()
                if outcome == 'failure':
                    return httpx.Response(200, content=b'odd')
            return httpx.Response(200, content=bytes(1024))
        async def emit(event):
            events.append(event)
            if event['type'] == 'audio_end':
                audio_end.set()
        async def send_audio(data):
            binary.append(data)
        async with asyncio.timeout(3), httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            bank = demo.DemoVoice(config, tmp_path, client=client)
            session = VoiceSession(config, tmp_path, object(), object(), bank, None, emit, send_audio)
            session.task.apply([{'op': 'set', 'slot': key, 'value': value} for key, value in values().items()])
            try:
                await session._choose(Action('readback'))
                await entered.wait()
                assert not binary and not any(event['type'] in {'audio_start', 'audio_end'} for event in events)
                assert session.task.readback_version is None
                if outcome == 'cancel':
                    await session.interrupt()
                    await bank.close()
                    assert len(requests) == 2
                else:
                    release.set()
                    if outcome == 'failure':
                        await session.playback_task
                        assert session.status == 'error'
                    else:
                        await audio_end.wait()
                        raw = b''.join(binary)
                        pcm = demo.pcm16(raw, config['runtime'])
                        assert len(pcm) == 1024 * len(requests)
                        assert len(raw) == next(event['bytes'] for event in events if event['type'] == 'audio_start')
                        assert session.task.readback_version is None
                        await session.playback_finished(session.playback_id)
                        await session.playback_task
                        assert session.task.readback_version == session.task.version
                if outcome != 'complete':
                    assert not binary
                    assert not any(event['type'] in {'audio_start', 'audio_end', 'playback_complete'} for event in events)
                    assert session.task.readback_version is None and not session.task.confirmed
                    whole = demo.reply_text(Action('readback'), values(), config['demo'])
                    assert not demo.speech_cache_path(config['demo'], tmp_path, whole).exists()
                    assert not list(tmp_path.rglob('*.tmp'))
            finally:
                await session.close()
                await bank.close()
    asyncio.run(run())
