"""Request-local ASR timing with mocked HTTP; no speech service is contacted."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

import engine.stt as stt_module
from engine.config import ROOT, load_config
from engine.stt import SpeechToText, STTError


@pytest.fixture
def config():
    return load_config(ROOT / 'configs/pizza.json')


def adapter(config, tmp_path, client):
    return SpeechToText(config, tmp_path, primary='moulsot', fallback=False,
                        endpoint='https://offline-fixture.hf.space', client=client,
                        hf_token='hf_never_record_this')


def test_gradio_phase_timings_use_perf_counter_and_contain_no_payloads(config, tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(stt_module, 'time', SimpleNamespace(perf_counter=lambda: clock[0]))
    seen = []

    def handle(request):
        seen.append(request)
        assert request.headers['Authorization'] == 'Bearer hf_never_record_this'
        if request.url.path.endswith('/upload'):
            clock[0] += .01
            return httpx.Response(200, json=['/tmp/private_audio.wav'])
        if request.method == 'POST':
            clock[0] += .02
            return httpx.Response(200, json={'event_id': 'abc123'})
        clock[0] += .03
        return httpx.Response(200, text='event: complete\ndata: ["private transcript"]\n\n')

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            stt = adapter(config, tmp_path, client)
            result = await stt.transcribe(bytes(1024))
            assert result.text == 'private transcript'
            record, = stt.calls
            assert record['elapsed_clock'] == 'perf_counter'
            assert record['pcm_bytes'] == 1024
            assert record['audio_duration_ms'] == 32
            assert record['phases_ms'] == pytest.approx({'upload': 10, 'submission': 20, 'result_wait': 30})
            assert record['elapsed_ms'] == pytest.approx(60)
            assert record['ok'] and record['confidence_kind'] == 'unavailable'
            assert 'not separated' in record['result_wait_basis']
            serialized = json.dumps(record)
            for private in ('hf_never_record_this', 'offline-fixture', 'private_audio', 'private transcript', 'abc123'):
                assert private not in serialized
            assert len(seen) == 3

    asyncio.run(run())


@pytest.mark.parametrize('failed_phase', ['upload', 'submission', 'result_wait'])
def test_failed_phase_and_overall_duration_are_recorded(config, tmp_path, failed_phase):
    phases = []

    def handle(request):
        phase = 'upload' if request.url.path.endswith('/upload') else 'submission' if request.method == 'POST' else 'result_wait'
        phases.append(phase)
        if phase == failed_phase:
            return httpx.Response(503, text='private provider body')
        if phase == 'upload':
            return httpx.Response(200, json=['/tmp/audio.wav'])
        return httpx.Response(200, json={'event_id': 'abc123'})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            stt = adapter(config, tmp_path, client)
            with pytest.raises(STTError):
                await stt.transcribe(bytes(1024))
            record, = stt.calls
            assert not record['ok'] and record['error'] == 'HTTPStatusError'
            assert list(record['phases_ms']) == phases
            assert all(duration >= 0 for duration in record['phases_ms'].values())
            assert record['elapsed_ms'] >= sum(record['phases_ms'].values())
            assert record['audio_duration_ms'] == 32
            assert 'private provider body' not in json.dumps(record)

    asyncio.run(run())


@pytest.mark.parametrize('outcome', ['timeout', 'cancel'])
def test_interrupted_stream_retains_phase_and_total_timing(config, tmp_path, outcome):
    async def run():
        entered = asyncio.Event()

        async def handle(request):
            if request.url.path.endswith('/upload'):
                return httpx.Response(200, json=['/tmp/audio.wav'])
            if request.method == 'POST':
                return httpx.Response(200, json={'event_id': 'abc123'})
            entered.set()
            await asyncio.Event().wait()

        config['stt']['timeout_ms'] = 10 if outcome == 'timeout' else 1000
        async with asyncio.timeout(2), httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            stt = adapter(config, tmp_path, client)
            task = asyncio.create_task(stt.transcribe(bytes(1024)))
            await entered.wait()
            if outcome == 'cancel':
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(STTError):
                    await task
            record, = stt.calls
            assert not record['ok']
            assert record['error'] == ('CancelledError' if outcome == 'cancel' else 'TimeoutError')
            assert record.get('cancelled', False) is (outcome == 'cancel')
            assert set(record['phases_ms']) == {'upload', 'submission', 'result_wait'}
            assert record['phases_ms']['result_wait'] > 0
            assert record['elapsed_ms'] >= sum(record['phases_ms'].values())
            assert record['pcm_bytes'] == 1024

    asyncio.run(run())


def test_concurrent_json_calls_keep_request_local_records(config, tmp_path):
    config['stt']['moulsot_protocol'] = 'json'

    async def run():
        first_entered, release_first = asyncio.Event(), asyncio.Event()
        requests = []

        async def handle(request):
            requests.append(request)
            assert 'Authorization' not in request.headers
            ordinal = len(requests)
            if ordinal == 1:
                first_entered.set()
                await release_first.wait()
            return httpx.Response(200, json={'text': f'private transcript {ordinal}', 'confidence': .9})

        async with asyncio.timeout(2), httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            stt = adapter(config, tmp_path, client)
            first = asyncio.create_task(stt.transcribe(bytes(1024)))
            await first_entered.wait()
            second = await stt.transcribe(bytes(2048))
            assert second.text == 'private transcript 2'
            assert len(stt.calls) == 1 and stt.calls[0]['pcm_bytes'] == 2048
            release_first.set()
            await first
            assert len(stt.calls) == 2
            records = {record['pcm_bytes']: record for record in stt.calls}
            assert records[1024]['audio_duration_ms'] == 32
            assert records[2048]['audio_duration_ms'] == 64
            assert records[1024]['phases_ms'] is not records[2048]['phases_ms']
            for record in records.values():
                assert record['ok'] and record['confidence_kind'] == 'provider_reported'
                assert set(record['phases_ms']) == {'json_request'}
                assert record['elapsed_ms'] >= record['phases_ms']['json_request'] >= 0
                assert 'private transcript' not in json.dumps(record)

    asyncio.run(run())
