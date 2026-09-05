"""Offline provider progress lifecycle for both domain previews."""

import asyncio
import json

import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.endpointing import Segment
from engine.pipeline import VoiceSession
from engine.responder import AudioBankError, RenderedAudio
from engine.router import RouterError
from engine.state import Action
from engine.stt import STTError, Transcript, wav_bytes


def make_session(domain, tmp_path, events, *, blocked=None, failure=None):
    config = demo_config(load_config(ROOT / 'configs' / f'{domain}.json'))
    gates = {stage: asyncio.Event() for stage in ('transcribing', 'routing', 'synthesizing')}
    entered = {stage: asyncio.Event() for stage in gates}
    exited = {stage: asyncio.Event() for stage in gates}

    async def phase(stage):
        entered[stage].set()
        try:
            if stage in (blocked or ()):
                await gates[stage].wait()
            if stage == failure:
                error = {'transcribing': STTError, 'routing': RouterError, 'synthesizing': AudioBankError}[stage]
                raise error('Offline provider fixture failure')
        finally:
            exited[stage].set()

    class ASR:
        async def transcribe(self, pcm):
            await phase('transcribing')
            return Transcript('offline request', .99, 'fixture')

    class Router:
        async def route(self, text, state):
            await phase('routing')
            return {'intent': 'task', 'ops': [], 'unclear': False,
                    'is_affirmation': False, 'is_negation': False}

    class Bank:
        async def render(self, action, values):
            if action.kind != 'greeting':
                await phase('synthesizing')
            return RenderedAudio(wav_bytes(bytes(1024), config['runtime']), 512, 512,
                                 ['offline-fixture'], 'Offline fixture reply')

    async def emit(event):
        events.append(dict(event))
        if event['type'] == 'audio_end':
            await voice.playback_finished(event['playback_id'])

    async def audio(data):
        pass

    voice = VoiceSession(config, tmp_path, ASR(), Router(), Bank(), None, emit, audio)
    return voice, gates, entered, exited


def progress(events):
    return [event for event in events if event['type'] == 'progress']


async def enqueue(voice):
    await voice.queue.put(Segment(bytes([1, 0]) * 512, 0, 320, 1728, 320))


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_progress_tracks_actual_awaited_stage_in_order(domain, tmp_path):
    async def run():
        events = []
        voice, gates, entered, _ = make_session(domain, tmp_path, events,
            blocked={'transcribing', 'routing', 'synthesizing'})
        async with asyncio.timeout(3):
            try:
                await voice.start()
                await voice.playback_task
                events.clear()
                await enqueue(voice)
                for stage in ('transcribing', 'routing', 'synthesizing'):
                    await entered[stage].wait()
                    active = progress(events)[-1]
                    assert active['stage'] == stage and active['active'] is True
                    assert isinstance(active['progress_id'], str) and active['progress_id']
                    gates[stage].set()
                await voice.queue.join()
                await voice.playback_task
                updates = progress(events)
                assert [(event['stage'], event['active']) for event in updates] == [
                    ('transcribing', True), ('transcribing', False),
                    ('routing', True), ('routing', False),
                    ('synthesizing', True), ('synthesizing', False),
                ]
                assert len({event['progress_id'] for event in updates[::2]}) == 3
                assert all(start['progress_id'] == end['progress_id']
                           for start, end in zip(updates[::2], updates[1::2]))
            finally:
                await voice.close()

    asyncio.run(run())


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
@pytest.mark.parametrize('failed_stage', ['transcribing', 'routing', 'synthesizing'])
def test_failed_stage_clears_progress_before_terminal_error(domain, tmp_path, failed_stage):
    async def run():
        events = []
        voice, _, _, _ = make_session(domain, tmp_path, events, failure=failed_stage)
        async with asyncio.timeout(3):
            try:
                await voice.start()
                await voice.playback_task
                events.clear()
                await enqueue(voice)
                await voice.done.wait()
                updates = [event for event in progress(events) if event['stage'] == failed_stage]
                assert [event['active'] for event in updates] == [True, False]
                assert updates[0]['progress_id'] == updates[1]['progress_id']
                assert events.index(updates[1]) < next(i for i, event in enumerate(events) if event['type'] == 'error')
                assert voice.status == 'error'
            finally:
                await voice.close()

    asyncio.run(run())


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_stop_cancels_blocked_transcription_without_late_events(domain, tmp_path):
    async def run():
        events = []
        voice, _, entered, exited = make_session(domain, tmp_path, events, blocked={'transcribing'})
        async with asyncio.timeout(3):
            await voice.start()
            await voice.playback_task
            await enqueue(voice)
            await entered['transcribing'].wait()
            assert progress(events)[-1]['active'] is True
            snapshot = list(events)
            await voice.close()
            assert voice.worker.done() and exited['transcribing'].is_set()
            assert events == snapshot
            assert not entered['routing'].is_set()
            saved, = tmp_path.rglob('demo_session_*.json')
            result = json.loads(saved.read_text(encoding='utf-8'))
            assert result['status'] == 'interrupted'
            assert result['session_id'] == voice.session_id
            assert not result['confirmed']

    asyncio.run(run())


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_barge_in_clears_cancelled_synthesis_and_next_reply_can_progress(domain, tmp_path):
    async def run():
        events = []
        voice, gates, entered, exited = make_session(domain, tmp_path, events, blocked={'synthesizing'})
        async with asyncio.timeout(3):
            try:
                await voice.start()
                await voice.playback_task
                events.clear()
                await enqueue(voice)
                await entered['synthesizing'].wait()
                cancelled_id = progress(events)[-1]['progress_id']
                await voice.interrupt()
                assert exited['synthesizing'].is_set()
                updates = [event for event in progress(events) if event['progress_id'] == cancelled_id]
                assert [event['active'] for event in updates] == [True, False]
                assert not voice.done.is_set()
                assert not any(event['type'] == 'error' for event in events)
                gates['synthesizing'].set()
                await voice._choose(Action('ask', 'size' if domain == 'pizza' else 'doctor'))
                await voice.playback_task
                assert progress(events)[-1]['active'] is False
                assert progress(events)[-1]['progress_id'] != cancelled_id
            finally:
                await voice.close()

    asyncio.run(run())


def test_old_progress_completion_cannot_clear_newer_progress(tmp_path):
    async def run():
        events = []
        voice, _, _, _ = make_session('pizza', tmp_path, events)
        old_started, new_started = asyncio.Event(), asyncio.Event()
        release_old, release_new = asyncio.Event(), asyncio.Event()

        async def hold(stage, entered, release):
            async with voice._progress(stage):
                entered.set()
                await release.wait()

        async with asyncio.timeout(3):
            old = asyncio.create_task(hold('transcribing', old_started, release_old))
            await old_started.wait()
            newer = asyncio.create_task(hold('synthesizing', new_started, release_new))
            await new_started.wait()
            newer_id = progress(events)[-1]['progress_id']
            before = len(events)
            release_old.set()
            await old
            assert not any(event['type'] == 'progress' and not event['active'] and
                           event['progress_id'] == newer_id for event in events[before:])
            release_new.set()
            await newer
            assert progress(events)[-1]['progress_id'] == newer_id
            assert progress(events)[-1]['active'] is False
            await voice.close()

    asyncio.run(run())
