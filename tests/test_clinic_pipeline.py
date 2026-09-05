"""Full clinic VoiceSession protocol with mocked providers, not speech accuracy."""

import asyncio
import io
import json
import wave

from engine.config import ROOT, load_config
from engine.demo import demo_config, reply_text
from engine.endpointing import Segment
from engine.pipeline import VoiceSession
from engine.responder import RenderedAudio
from engine.router import parse_response
from engine.scoped_task import ScopedTaskState
from engine.stt import Transcript, wav_bytes


def test_clinic_proposal_correction_and_complete_audio_frames(tmp_path):
    config = demo_config(load_config(ROOT / 'configs/clinic.json'))

    def operation(slot, value): return dict(op='set', slot=slot, value=value)

    class ASR:
        turn = 0
        async def transcribe(self, pcm):
            self.turn += 1
            return Transcript(f'canonical fixture turn {self.turn}', .99, 'mock_asr')

    class Route:
        turn = 0
        async def route(self, text, state):
            self.turn += 1
            result = dict(ops=[], confidence=.99, unclear=False, is_affirmation=False,
                          is_negation=False, intent='task', clarification=None,
                          resolves_clarification=None, proposed_ops=[], discard_clarification=None)
            if self.turn == 1:
                result.update(intent='out_of_scope',
                    clarification=dict(kind='unsupported_value', slot='doctor', item_ids=[]),
                    proposed_ops=[operation('date', '2026-09-15'), operation('time', '09:30')])
            elif self.turn == 2:
                assert state['state'] == {} and state['pending_request'] is not None
                assert state['pending_proposal']['state'] == {'date': '2026-09-15', 'time': '09:30'}
                result.update(ops=[operation('doctor', 'doctor_b')],
                              resolves_clarification=state['pending_clarification']['id'])
            elif self.turn == 3:
                assert state['pending_request'] is None and state['readback_complete']
                result.update(ops=[operation('time', '10:30')], is_affirmation=True)
            else:
                assert self.turn == 4 and state['readback_complete']
                result['is_affirmation'] = True
            return parse_response(json.dumps(result), config)

    class Bank:
        async def render(self, action, values):
            pcm = bytes(7000)  # More than one WebSocket-sized frame.
            return RenderedAudio(wav_bytes(pcm, config['runtime']), 3500, 3500,
                                 ['silent fixture'], reply_text(action, values, config['demo']))

    async def run():
        events, packets = [], {}
        active = None

        async def emit(event):
            nonlocal active
            events.append(event)
            if event['type'] == 'audio_start':
                active = event['playback_id']
                packets[active] = {'expected': event['bytes'], 'parts': []}
            elif event['type'] == 'audio_end':
                packet = packets[event['playback_id']]
                data = b''.join(packet['parts'])
                assert len(packet['parts']) > 1 and len(data) == packet['expected']
                with wave.open(io.BytesIO(data), 'rb') as audio:
                    assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 16000)
                    assert len(audio.readframes(audio.getnframes())) == 7000
                await voice.playback_finished(event['playback_id'])

        async def audio(data): packets[active]['parts'].append(data)

        voice = VoiceSession(config, tmp_path, ASR(), Route(), Bank(), None, emit, audio)
        assert isinstance(voice.task, ScopedTaskState)
        try:
            await voice.start()
            await voice.playback_task
            for turn in range(1, 5):
                await voice.queue.put(Segment(bytes([turn, 0]) * 512, 0, 320, 1728, 320))
                await voice.queue.join()
                await voice.playback_task
                if turn == 1:
                    assert voice.task.values == {} and voice.task.pending_proposal is not None
                if turn == 3:
                    assert not voice.task.confirmed and voice.task.values['time'] == '10:30'
            assert voice.status == 'demo_completed' and voice.task.confirmed
            assert voice.task.values == {'date': '2026-09-15', 'time': '10:30', 'doctor': 'doctor_b'}
            assert not any(event['type'] in {'error', 'warning'} for event in events)
        finally:
            await voice.close()
        saved = json.loads(next(tmp_path.glob('bench/results/demo_session_*.json')).read_text(encoding='utf-8'))
        assert saved['domain'] == 'clinic' and saved['evaluation_eligible'] is False
        assert 'no availability check or booking' in saved['task_scope']
        assert saved['pending_proposal'] is None and saved['turns'][0]['state'] == {}
        assert [output['action'] for output in saved['outputs']] == [
            'greeting', 'unsupported_value', 'readback', 'readback', 'accepted']
        assert all(output['completed'] for output in saved['outputs'])

    asyncio.run(run())
