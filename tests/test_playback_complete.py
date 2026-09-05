"""Exact playback completion is emitted only for the currently acknowledged audio."""

import asyncio
from copy import deepcopy

import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.pipeline import VoiceSession
from engine.responder import RenderedAudio
from engine.state import Action
from engine.stt import wav_bytes


def make_voice(domain, root, events, ended):
    config = demo_config(load_config(ROOT / 'configs' / f'{domain}.json'))

    class Bank:
        async def render(self, action, values):
            return RenderedAudio(wav_bytes(bytes(1024), config['runtime']), 512, 512,
                                 ['offline fixture'], 'Offline fixture')

    async def emit(event):
        events.append(deepcopy(event))
        if event['type'] == 'audio_end':
            ended.set()

    async def emit_audio(raw):
        pass

    voice = VoiceSession(config, root, None, None, Bank(), None, emit, emit_audio)
    if domain == 'clinic':
        voice.task.apply([{'op': 'set', 'slot': key, 'value': value} for key, value in
                          {'doctor': 'doctor_a', 'date': '2026-09-15', 'time': '10:30'}.items()])
    else:
        voice.task.apply([{'op': 'create', 'item_id': 1, 'slot': None, 'value': None},
                          *[{'op': 'set', 'item_id': None if key == 'drink' else 1,
                             'slot': key, 'value': value} for key, value in
                            {'quantity': 1, 'size': 'large', 'toppings': [], 'drink': ['none']}.items()]])
    return voice


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_exact_completion_follows_matching_ack_and_confirmation(domain, tmp_path):
    async def run():
        events, ended = [], asyncio.Event()
        voice = make_voice(domain, tmp_path, events, ended)
        try:
            await voice._choose(Action('readback'))
            await asyncio.wait_for(ended.wait(), 1)
            playback_id = voice.playback_id
            await voice.playback_finished('stale-playback-id')
            assert not voice.playback_done.is_set()
            assert not any(event['type'] == 'playback_complete' for event in events)
            await voice.playback_finished(playback_id)
            await asyncio.wait_for(voice.playback_task, 1)
            await voice.playback_finished(playback_id)
            completed = [event for event in events if event['type'] == 'playback_complete']
            assert len(completed) == 1
            result = completed[0]
            assert result['playback_id'] == playback_id and result['action'] == 'readback'
            assert result['slots'] == voice.task.values
            assert result['version'] == result['spoken_version'] == voice.task.version
            assert result['readback_complete'] is True
            assert not voice.task.confirmed
            assert events[-1]['type'] == 'state' and events[-1]['state'] == 'CONFIRMING'
        finally:
            await voice.close()
    asyncio.run(run())


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_interrupted_audio_never_emits_completion(domain, tmp_path):
    async def run():
        events, ended = [], asyncio.Event()
        voice = make_voice(domain, tmp_path, events, ended)
        try:
            await voice._choose(Action('readback'))
            await asyncio.wait_for(ended.wait(), 1)
            old_id = voice.playback_id
            await voice.interrupt()
            await voice.playback_finished(old_id)
            assert not any(event['type'] == 'playback_complete' for event in events)
            assert voice.task.readback_version is None
        finally:
            await voice.close()
    asyncio.run(run())


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_completion_precedes_terminal_result(domain, tmp_path):
    async def run():
        events, ended = [], asyncio.Event()
        voice = make_voice(domain, tmp_path, events, ended)
        try:
            await voice._choose(Action('handoff'))
            await asyncio.wait_for(ended.wait(), 1)
            await voice.playback_finished(voice.playback_id)
            await asyncio.wait_for(voice.playback_task, 1)
            kinds = [event['type'] for event in events]
            assert kinds.index('playback_complete') < kinds.index('done')
            assert voice.status == 'handoff' and voice.done.is_set()
        finally:
            await voice.close()
    asyncio.run(run())
