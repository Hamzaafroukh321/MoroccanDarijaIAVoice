"""Offline playback lifecycle checks across both synthetic domain previews."""

import asyncio
import json
from unittest.mock import patch

import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.pipeline import TurnState, VoiceSession
from engine.responder import RenderedAudio
from engine.state import Action
from engine.stt import wav_bytes


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
@pytest.mark.parametrize('previous_tail_frames', [0, 2], ids=['fresh', 'old_voiced_tail'])
def test_barge_in_counts_only_frames_from_current_playback(tmp_path, domain, previous_tail_frames):
    config = demo_config(load_config(ROOT / 'configs' / f'{domain}.json'))

    class Bank:
        async def render(self, action, values):
            return RenderedAudio(wav_bytes(bytes(1024), config['runtime']), 512, 512,
                                 ['offline-fixture'], 'Offline playback fixture')

    async def run():
        audio_ends = asyncio.Queue()
        events = []

        async def emit(event):
            events.append(event)
            if event['type'] == 'audio_end':
                audio_ends.put_nowait(event['playback_id'])

        async def audio(data):
            pass

        voice = VoiceSession(config, tmp_path, None, None, Bank(), lambda pcm: 1.0, emit, audio)
        async with asyncio.timeout(3):
            try:
                await voice.start()
                greeting_id = await audio_ends.get()
                old_frame = bytes([1, 0]) * 512
                for _ in range(previous_tail_frames):
                    await voice.feed(old_frame)
                assert voice.playback_id == greeting_id
                await voice.playback_finished(greeting_id)
                await voice.playback_task
                assert voice.mode == TurnState.LISTENING

                slot = 'size' if domain == 'pizza' else 'doctor'
                await voice._choose(Action('ask', slot))
                reply_id = await audio_ends.get()
                fresh_frames = [bytes([number, 0]) * 512 for number in (2, 3, 4)]
                await voice.feed(fresh_frames[0])
                assert voice.playback_id == reply_id, 'A single fresh frame must not reuse a previous playback tail.'
                assert not voice.detector.active
                for frame in fresh_frames[1:]:
                    await voice.feed(frame)
                assert voice.playback_id is None
                assert voice.mode == TurnState.LISTENING
                assert voice.detector.speech_ms == 96
                assert bytes(voice.detector.buffer) == b''.join(fresh_frames)
                assert [event['playback_id'] for event in events if event['type'] == 'audio_stop'] == [reply_id]
            finally:
                await voice.close()

    asyncio.run(run())


@pytest.mark.parametrize('cancel_first_close', [False, True], ids=['concurrent_close', 'cancel_then_retry'])
def test_session_close_waits_for_cleanup_and_saves_once(tmp_path, cancel_first_close):
    config = demo_config(load_config(ROOT / 'configs/pizza.json'))

    async def run():
        entered = asyncio.Event()
        cleanup_started = asyncio.Event()
        cleanup_release = asyncio.Event()
        cleanup_finished = asyncio.Event()

        async def worker():
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_started.set()
                await cleanup_release.wait()
                cleanup_finished.set()
                raise

        async def emit(event):
            pass

        voice = VoiceSession(config, tmp_path, None, None, None, None, emit, emit)
        voice.worker = asyncio.create_task(worker())
        original_save = voice.save

        def save_after_cleanup():
            assert cleanup_finished.is_set()
            return original_save()

        async with asyncio.timeout(3):
            with patch.object(voice, 'save', side_effect=save_after_cleanup) as save:
                await entered.wait()
                first = asyncio.create_task(voice.close())
                await cleanup_started.wait()
                if cancel_first_close:
                    first.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await first
                second = asyncio.create_task(voice.close())
                await asyncio.sleep(0)
                assert not second.done(), 'Every close caller must wait for cancellation cleanup and persistence.'
                assert not cleanup_finished.is_set()
                assert voice.closed
                assert save.call_count == 0
                assert not list(tmp_path.rglob('demo_session_*.json'))
                cleanup_release.set()
                await second
                if not cancel_first_close:
                    await first
                assert cleanup_finished.is_set()
                assert voice.worker.done()
                await voice.close()
                assert save.call_count == 1
                result_path, = tmp_path.rglob('demo_session_*.json')
                result = json.loads(result_path.read_text(encoding='utf-8'))
                assert result['session_id'] == voice.session_id
                assert result['status'] == 'interrupted'
                assert not result['confirmed']

    asyncio.run(run())
