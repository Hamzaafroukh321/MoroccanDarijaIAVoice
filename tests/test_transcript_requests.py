"""Track actual calls versus cached transcript waiters without changing ASR."""
import asyncio
import json

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.pipeline import VoiceSession
from engine.stt import Transcript


def test_partial_and_final_share_one_asr_call_and_save_distinct_waiters(tmp_path):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        class STT:
            count = 0
            async def transcribe(self, pcm):
                self.count += 1
                entered.set()
                await release.wait()
                return Transcript('fixture', None, 'mock')
        async def emit(value): pass
        stt = STT()
        session = VoiceSession(demo_config(load_config(ROOT/'configs/clinic.json')),
            tmp_path, stt, None, None, None, emit, emit)
        pcm = bytes(1024)
        partial = asyncio.create_task(session._transcript(pcm, purpose='partial'))
        await entered.wait()
        final = asyncio.create_task(session._transcript(pcm))
        await asyncio.sleep(0)
        partial.cancel()
        try: await partial
        except asyncio.CancelledError: pass
        assert not session.cache[next(iter(session.cache))].cancelled()
        release.set()
        assert (await final).text == 'fixture'
        assert stt.count == 1
        await session.close()
        result = json.loads(next(tmp_path.rglob('demo_session_*.json')).read_text(encoding='utf-8'))
        first, second = result['transcript_requests']
        assert first['purpose'] == 'partial' and first['cancelled'] and not first['ok']
        assert not first['reused_transcription']
        assert second['purpose'] == 'final' and second['ok'] and second['reused_transcription']
        assert first['audio_sha256'] == second['audio_sha256']
        assert second['pcm_bytes'] == 1024 and second['audio_seconds'] == .032
        assert all(row['wait_ms'] >= 0 for row in (first, second))
    asyncio.run(run())
