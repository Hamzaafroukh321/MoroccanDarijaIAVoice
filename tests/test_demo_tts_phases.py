"""Offline XTTS subphase evidence distinguishes render waits from producer work."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest

import engine.demo as demo_module
from engine.config import ROOT, load_config
from engine.demo import DemoVoice, demo_config, pcm16
from engine.responder import AudioBankError
from engine.pipeline import VoiceSession
from engine.state import Action
from engine.stt import wav_bytes


SECRET = 'PRIVATE_FIXTURE_TOKEN'


class Clock:
    def __init__(self):
        self.value = 10.0

    def perf_counter(self):
        self.value += .000125
        return self.value

    def monotonic(self):
        return 10.0


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    monkeypatch.setenv('HF_TOKEN', SECRET)
    value = demo_config(load_config(ROOT / 'configs/pizza.json'))
    value['demo']['cache_dir'] = 'cache'
    value['demo']['responses']['greeting'] = SECRET
    monkeypatch.setattr(demo_module, 'time', Clock())
    return value


class XTTS:
    def __init__(self, config, *, blocked=None, fail=None):
        self.config, self.blocked, self.fail = config, blocked, fail
        self.seen = []
        self.entered, self.release, self.cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def handle(self, request):
        path = request.url.path
        phase = ('reference' if path == '/config' else 'submission' if request.method == 'POST'
            else 'download' if '/file=' in path else 'wait')
        self.seen.append(phase)
        demo_module.time.value += {'reference': .02, 'submission': .03, 'wait': .05, 'download': .07}[phase]
        if self.blocked == phase:
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        if self.fail == phase:
            return httpx.Response(503, text=SECRET)
        if phase == 'reference':
            return httpx.Response(200, json={'components': [{'type': 'audio', 'props': {
                'label': 'Speaker reference', 'value': {'path': '/tmp/gradio/private-reference.wav'}}}]})
        if phase == 'submission':
            return httpx.Response(200, json={'event_id': 'job123'})
        if phase == 'wait':
            return httpx.Response(200, text='event: complete\ndata: '
                + json.dumps([{'path': '/tmp/gradio/private-result.wav'}]) + '\n\n')
        return httpx.Response(200, content=wav_bytes(bytes(4800), dict(self.config['runtime'], sample_rate_hz=24000)))


def safe_metadata(voice):
    text = json.dumps({'renders': voice.calls, 'producers': voice.producer_calls})
    for private in (SECRET, 'private-reference', 'private-result', 'hf.space', 'job123', 'Authorization'):
        assert private not in text
    for record in [*voice.calls, *voice.producer_calls]:
        assert record['elapsed_clock'] == 'perf_counter'
        if record.get('status') != 'in_progress':
            assert record['elapsed_ms'] > 0
        for duration in record['phases_ms'].values():
            assert isinstance(duration, (int, float)) and duration >= 0


def test_cold_render_distinguishes_remote_wait_download_and_local_conversion_then_hits_cache(config, tmp_path):
    async def run():
        upstream = XTTS(config)
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream.handle)) as client:
            voice = DemoVoice(config, tmp_path, client=client)
            first = await voice.render(Action('greeting'), {})
            assert len(pcm16(first.wav, config['runtime'])) == 3200
            producer, = voice.producer_calls
            assert producer['status'] == 'completed'
            for key, lower, upper in [('reference_lookup', 20, 21), ('api_submission', 30, 31),
                ('remote_wait', 50, 51), ('result_download', 70, 71)]:
                assert lower <= producer['phases_ms'][key] < upper
            assert producer['phases_ms']['audio_validation'] > 0
            assert producer['phases_ms']['cache_write'] > 0
            render, = voice.calls
            assert producer['owner_render_id'] == render['render_id']
            part, = render['part_timings']
            assert part['producer_id'] == producer['producer_id'] and part['cache_outcome'] == 'miss'
            assert part['phases_ms']['cache_read_validate'] > 0
            assert part['phases_ms']['pcm_conversion'] > 0
            assert render['phases_ms']['assembly'] > 0
            assert 'remote_wait' not in part['phases_ms']
            assert upstream.seen == ['reference', 'submission', 'wait', 'download']
            second = await voice.render(Action('greeting'), {})
            assert second.wav == first.wav and len(voice.producer_calls) == 1
            hit = voice.calls[-1]
            assert hit['render_id'] != render['render_id']
            assert hit['cache_hits'] == 1 and hit['cache_misses'] == 0
            assert hit['part_timings'][0]['cache_outcome'] == 'hit'
            assert not hit['part_timings'][0].get('producer_id')
            assert set(hit['part_timings'][0]['phases_ms']) <= {'cache_read_validate', 'pcm_conversion'}
            assert upstream.seen == ['reference', 'submission', 'wait', 'download']
            safe_metadata(voice)
            await voice.close()
    asyncio.run(run())


@pytest.mark.parametrize('cancel_owner', [False, True])
def test_shared_waiters_reference_one_producer_and_cancelled_render_record_is_final(config, tmp_path, cancel_owner):
    async def run():
        upstream = XTTS(config, blocked='wait')
        async with asyncio.timeout(3), httpx.AsyncClient(transport=httpx.MockTransport(upstream.handle)) as client:
            voice = DemoVoice(config, tmp_path, client=client)
            owner = asyncio.create_task(voice.render(Action('greeting'), {}))
            await upstream.entered.wait()
            waiter = asyncio.create_task(voice.render(Action('greeting'), {}))
            await asyncio.sleep(0)
            cancelled_record = None
            if cancel_owner:
                owner.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await owner
                cancelled_record = deepcopy(voice.calls[0])
                assert not cancelled_record['ok']
                assert cancelled_record['elapsed_ms'] > 0
                assert cancelled_record['part_timings'][0]['phases_ms']['inflight_wait'] > 0
                assert voice.producer_calls[0]['status'] == 'in_progress'
                assert not upstream.cancelled.is_set()
            upstream.release.set()
            result = await waiter
            if not cancel_owner:
                assert (await owner).wav == result.wav
            assert upstream.seen.count('submission') == 1
            assert len(voice.producer_calls) == 1 and voice.producer_calls[0]['status'] == 'completed'
            assert len(voice.calls) == 2
            assert sum(record['cache_misses'] for record in voice.calls) == 1
            assert sum(record['coalesced_waits'] for record in voice.calls) == 1
            assert {record['part_timings'][0]['producer_id'] for record in voice.calls} == {voice.producer_calls[0]['producer_id']}
            assert not any('remote_wait' in record['phases_ms'] for record in voice.calls)
            if cancelled_record:
                assert voice.calls[0] == cancelled_record
            safe_metadata(voice)
            await voice.close()
    asyncio.run(run())


@pytest.mark.parametrize('failed_phase,expected', [('reference', 'reference_lookup'),
    ('submission', 'api_submission'), ('wait', 'remote_wait'), ('download', 'result_download')])
def test_failed_producer_finalizes_phase_and_total_without_publishing_audio(config, tmp_path, failed_phase, expected):
    async def run():
        upstream = XTTS(config, fail=failed_phase)
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream.handle)) as client:
            voice = DemoVoice(config, tmp_path, client=client)
            with pytest.raises(AudioBankError) as error:
                await voice.render(Action('greeting'), {})
            assert SECRET not in str(error.value)
            producer, = voice.producer_calls
            assert producer['status'] == 'failed' and producer['failure_phase'] == expected
            assert producer['phases_ms'][expected] > 0 and producer['elapsed_ms'] > 0
            assert not voice.calls[0]['ok'] and voice.calls[0]['elapsed_ms'] > 0
            assert not list(tmp_path.rglob('*.wav'))
            safe_metadata(voice)
            await voice.close()
    asyncio.run(run())


def test_close_during_remote_wait_finalizes_cancelled_producer_and_render(config, tmp_path):
    async def run():
        upstream = XTTS(config, blocked='wait')
        async with asyncio.timeout(3), httpx.AsyncClient(transport=httpx.MockTransport(upstream.handle)) as client:
            voice = DemoVoice(config, tmp_path, client=client)
            rendering = asyncio.create_task(voice.render(Action('greeting'), {}))
            await upstream.entered.wait()
            await voice.close()
            with pytest.raises(asyncio.CancelledError):
                await rendering
            assert upstream.cancelled.is_set()
            producer, = voice.producer_calls
            assert producer['status'] == 'cancelled' and producer['failure_phase'] == 'remote_wait'
            assert producer['phases_ms']['remote_wait'] > 0
            assert not voice.calls[0]['ok']
            assert not list(tmp_path.rglob('*.wav'))
            assert not list(tmp_path.rglob('*.tmp'))
            safe_metadata(voice)
    asyncio.run(run())


def test_session_save_snapshots_only_its_producers_without_finalizing_inflight_work(config, tmp_path):
    bank = SimpleNamespace(calls=[{'render_id': 'earlier'}], producer_calls=[{'producer_id': 'earlier'}])
    voice = VoiceSession(config, tmp_path, SimpleNamespace(), SimpleNamespace(), bank, None, None, None)
    producer = {'producer_id': 'producer_2', 'owner_render_id': 'render_2',
        'status': 'in_progress', 'elapsed_clock': 'perf_counter', 'phases_ms': {'api_submission': 3}}
    render = {'render_id': 'render_2', 'ok': False, 'elapsed_ms': 5, 'elapsed_clock': 'perf_counter',
        'part_timings': [{'producer_id': 'producer_2', 'phases_ms': {'inflight_wait': 5}}]}
    bank.producer_calls.append(producer)
    bank.calls.append(render)
    snapshot = voice.save()
    path, = tmp_path.rglob('demo_session_*.json')
    persisted = path.read_text(encoding='utf-8')
    assert snapshot['tts_producer_calls'] == [producer] and snapshot['tts_calls'] == [render]
    assert 'elapsed_ms' not in snapshot['tts_producer_calls'][0]
    producer.update(status='completed', elapsed_ms=8)
    producer['phases_ms']['remote_wait'] = 4
    render['part_timings'][0]['phases_ms']['inflight_wait'] = 999
    assert snapshot['tts_producer_calls'][0]['status'] == 'in_progress'
    assert snapshot['tts_producer_calls'][0]['phases_ms'] == {'api_submission': 3}
    assert snapshot['tts_calls'][0]['part_timings'][0]['phases_ms']['inflight_wait'] == 5
    assert path.read_text(encoding='utf-8') == persisted


def test_multiple_parts_keep_distinct_producer_ownership_and_assemble_once(config, tmp_path):
    config['demo']['tts_max_chars'] = 32
    text = 'First synthetic sentence. Second synthetic sentence.'
    config['demo']['responses']['greeting'] = text

    async def run():
        upstream = XTTS(config)
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream.handle)) as client:
            voice = DemoVoice(config, tmp_path, client=client)
            result = await voice.render(Action('greeting'), {})
            render, = voice.calls
            assert result.text == text and result.total_samples == 3200
            assert render['parts'] == 2 and render['phases_ms']['assembly'] > 0
            assert [part['index'] for part in render['part_timings']] == [0, 1]
            assert len(voice.producer_calls) == 2
            for index, producer in enumerate(voice.producer_calls):
                assert producer['owner_render_id'] == render['render_id']
                assert producer['owner_part_index'] == index
                assert producer['producer_id'] == render['part_timings'][index]['producer_id']
            assert voice.producer_calls[0]['producer_id'] != voice.producer_calls[1]['producer_id']
            assert upstream.seen.count('reference') == 1
            assert upstream.seen.count('submission') == 2
            assert 'reference_lookup' not in voice.producer_calls[1]['phases_ms']
            assert text not in json.dumps(voice.calls + voice.producer_calls)
            safe_metadata(voice)
            await voice.close()
    asyncio.run(run())
