"""Mocked voice-session proposal transactions; no ASR accuracy claims or network."""

import asyncio
from copy import deepcopy
import json

import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config, reply_text
from engine.endpointing import Segment
from engine.pipeline import VoiceSession
from engine.responder import RenderedAudio
from engine.stt import Transcript, wav_bytes


def operation(kind, item_id=None, slot=None, value=None):
    return dict(op=kind, item_id=item_id, slot=slot, value=value)


@pytest.mark.parametrize('outcome', ['complete', 'incomplete', 'discard'])
def test_staged_proposal_voice_transaction_clears_original_request_and_requires_readback(tmp_path, outcome):
    config = demo_config(load_config(ROOT / 'configs/pizza.json'))
    original = 'one large cheese pizza and one medium beef pizza with a large bottle of cola'
    contexts = []
    proposed = [operation('create', 1), operation('set', 1, 'quantity', 1),
                operation('set', 1, 'size', 'large'), operation('set', 1, 'toppings', ['cheese']),
                operation('create', 2), operation('set', 2, 'quantity', 1),
                operation('set', 2, 'size', 'medium')]
    if outcome != 'incomplete':
        proposed.append(operation('set', 2, 'toppings', ['beef']))

    class ASR:
        text = None

        async def transcribe(self, pcm):
            return Transcript(self.text, .99, 'fixture')

    class Route:
        async def route(self, text, state):
            contexts.append((text, deepcopy(state)))
            response = dict(intent='task', ops=[], unclear=False, is_affirmation=False,
                            is_negation=False, clarification=None, resolves_clarification=None,
                            proposed_ops=[], discard_clarification=None)
            if text == original:
                response.update(intent='out_of_scope',
                                clarification=dict(kind='drink_size', slot='drink', item_ids=[]),
                                proposed_ops=deepcopy(proposed))
            elif text in {'cola', 'cancel'}:
                assert state['state'] == {'items': []}
                assert state['pending_request']['text'] == original
                assert state['pending_proposal']['next_item_id'] == 3
                assert len(state['pending_proposal']['state']['items']) == 2
                if text == 'cancel':
                    response['discard_clarification'] = state['pending_clarification']['id']
                else:
                    response.update(resolves_clarification=state['pending_clarification']['id'],
                                    ops=[operation('set', None, 'drink', ['cola'])])
            elif text == 'finish':
                assert state['pending_request'] is None
                assert state['pending_clarification'] is None and state['pending_proposal'] is None
                assert state['requested_item_id'] == 2 and state['requested_slot'] == 'toppings'
                response['ops'] = [operation('set', 2, 'toppings', ['beef'])]
            else:
                assert text == 'yes'
                assert state['pending_request'] is None
                assert state['pending_clarification'] is None and state['pending_proposal'] is None
                response['is_affirmation'] = True
            return response

    class Bank:
        async def render(self, action, values):
            return RenderedAudio(wav_bytes(bytes(1024), config['runtime']), 512, 512,
                                 ['fixture'], reply_text(action, values, config['demo']))

    async def run():
        events = []
        actions = {}
        readbacks = asyncio.Queue()
        asr = ASR()

        async def emit(event):
            events.append(deepcopy(event))
            if event['type'] == 'audio_start':
                actions[event['playback_id']] = event['action']
            elif event['type'] == 'audio_end':
                if actions[event['playback_id']] == 'readback':
                    readbacks.put_nowait(event['playback_id'])
                else:
                    await voice.playback_finished(event['playback_id'])

        async def audio(data):
            pass

        voice = VoiceSession(config, tmp_path, asr, Route(), Bank(), None, emit, audio)
        turn_count = 0

        async def speak(text, *, readback=False):
            nonlocal turn_count
            turn_count += 1
            asr.text = text
            # Every turn has a different PCM cache key, without real microphone I/O.
            await voice.queue.put(Segment(bytes([turn_count, 0]) * 512, 0, 320, 1728, 320))
            await voice.queue.join()
            if readback:
                return await asyncio.wait_for(readbacks.get(), 1)
            if voice.playback_task:
                await asyncio.wait_for(voice.playback_task, 1)
            return None

        try:
            await voice.start()
            await voice.playback_task
            greeting_id = next(event['playback_id'] for event in events if event['type'] == 'audio_start')
            await speak(original)
            assert voice.task.values == {'items': []}
            assert voice.task.next_item_id == 1 and voice.task.version == 0
            assert voice.pending_request['text'] == original
            assert voice.task.pending_proposal is not None and not voice.task.confirmed
            assert voice.turns[0]['state'] == {'items': []}

            readback_id = await speak('cancel' if outcome == 'discard' else 'cola', readback=outcome == 'complete')
            assert voice.pending_request is None
            assert voice.task.pending_clarification is None and voice.task.pending_proposal is None
            assert not voice.task.confirmed
            # Stored snapshots must not later acquire the proposal's committed rows.
            assert voice.turns[0]['state'] == {'items': []}
            assert voice.turns[1]['router_context']['state'] == {'items': []}
            assert len(voice.turns[1]['router_context']['pending_proposal']['state']['items']) == 2

            if outcome == 'discard':
                assert voice.task.values == {'items': []} and voice.task.next_item_id == 1
                await voice.playback_finished(greeting_id)
                await speak('yes')
                assert not voice.task.ready and not voice.task.confirmed
                assert all(output['action'] != 'readback' for output in voice.outputs)
            else:
                assert [row['id'] for row in voice.task.values['items']] == [1, 2]
                assert voice.task.next_item_id == 3 and voice.task.values['drink'] == ['cola']
                if outcome == 'incomplete':
                    assert not voice.task.ready and 'toppings' not in voice.task.values['items'][1]
                    readback_id = await speak('finish', readback=True)
                assert voice.task.ready and voice.task.readback_version is None
                # Neither a stale ACK nor bare yes during an unfinished readback
                # can confirm the order. A new completed readback is required.
                await voice.playback_finished(greeting_id)
                assert not voice.playback_done.is_set()
                replacement_id = await speak('yes', readback=True)
                assert replacement_id != readback_id
                assert not voice.task.confirmed and voice.task.readback_version is None
                assert contexts[-1][1]['readback_complete'] is False
                await voice.playback_finished(readback_id)
                assert not voice.playback_done.is_set()
                await voice.playback_finished(replacement_id)
                await voice.playback_task
                assert voice.task.readback_version == voice.task.version
                await speak('yes')
                assert voice.task.confirmed and voice.status == 'demo_completed'
                assert contexts[-1][1]['readback_complete'] is True
                assert [row['id'] for row in voice.task.values['items']] == [1, 2]
                assert voice.task.next_item_id == 3
        finally:
            await voice.close()

        saved = json.loads(next(tmp_path.glob('bench/results/demo_session_*.json')).read_text(encoding='utf-8'))
        assert saved['turns'][0]['state'] == {'items': []}
        assert saved['turns'][1]['router_context']['pending_request']['text'] == original
        assert saved['turns'][1]['router_context']['state'] == {'items': []}
        assert saved['pending_proposal'] is None and saved['pending_clarification'] is None
        assert saved['confirmed'] is (outcome != 'discard')
        assert saved['evaluation_eligible'] is False

    asyncio.run(run())
