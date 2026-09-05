"""Synthetic speech cache concurrency tests using only mocked HTTP responses."""

import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import httpx
import pytest

from engine.config import ROOT, load_config
from engine.demo import DemoVoice, demo_config, pcm16, speech_cache_path
from engine.responder import AudioBankError
from engine.state import Action


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'elevenlabs')
    monkeypatch.setenv('ELEVENLABS_API_KEY', 'offline-fixture')
    config = demo_config(load_config(ROOT / 'configs/pizza.json'))
    config['demo']['cache_dir'] = 'cache'
    return config


def test_identical_concurrent_misses_submit_one_provider_job(demo, tmp_path):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        requests = []

        async def handle(request):
            requests.append(request)
            entered.set()
            await release.wait()
            return httpx.Response(200, content=bytes(1024))

        async with asyncio.timeout(3), httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            voice = DemoVoice(demo, tmp_path, client=client)
            first = asyncio.create_task(voice.render(Action('greeting'), {}))
            await entered.wait()
            second = asyncio.create_task(voice.render(Action('greeting'), {}))
            await asyncio.sleep(0)
            assert not list(tmp_path.rglob('*.wav'))
            release.set()
            a, b = await asyncio.gather(first, second)
            assert a.wav == b.wav
            assert len(requests) == 1
            assert sum(call['cache_misses'] for call in voice.calls) == 1
            assert sum(call['coalesced_waits'] for call in voice.calls) == 1
            assert sum(call['cache_hits'] for call in voice.calls) == 0
            assert all(call['ok'] for call in voice.calls)
            await voice.render(Action('greeting'), {})
            assert voice.calls[-1]['cache_hits'] == 1
            assert len(requests) == 1
            await voice.close()
            assert not client.is_closed

    asyncio.run(run())


def test_cancelled_consumer_does_not_cancel_shared_generation(demo, tmp_path):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        requests = []
        producer_cancelled = []

        async def handle(request):
            requests.append(request)
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                producer_cancelled.append(True)
                raise
            return httpx.Response(200, content=bytes(1024))

        async with asyncio.timeout(3), httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            voice = DemoVoice(demo, tmp_path, client=client)
            first = asyncio.create_task(voice.render(Action('greeting'), {}))
            await entered.wait()
            second = asyncio.create_task(voice.render(Action('greeting'), {}))
            await asyncio.sleep(0)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert not producer_cancelled
            release.set()
            rendered = await second
            assert pcm16(rendered.wav, demo['runtime']) == bytes(1024)
            assert len(requests) == 1
            assert len(list(tmp_path.rglob('*.wav'))) == 1
            assert sorted(call['ok'] for call in voice.calls) == [False, True]
            await voice.close()

    asyncio.run(run())


def test_failed_shared_job_publishes_nothing_and_can_retry(demo, tmp_path):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def handle(request):
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return httpx.Response(200, content=b'odd' if calls == 1 else bytes(1024))

        async with asyncio.timeout(3), httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            voice = DemoVoice(demo, tmp_path, client=client)
            first = asyncio.create_task(voice.render(Action('greeting'), {}))
            await entered.wait()
            second = asyncio.create_task(voice.render(Action('greeting'), {}))
            await asyncio.sleep(0)
            release.set()
            results = await asyncio.gather(first, second, return_exceptions=True)
            assert all(isinstance(result, AudioBankError) for result in results)
            assert calls == 1
            assert not [path for path in tmp_path.rglob('*') if path.is_file()]
            rendered = await voice.render(Action('greeting'), {})
            assert rendered.total_samples == 512
            assert calls == 2
            assert len(list(tmp_path.rglob('*.wav'))) == 1
            await voice.close()

    asyncio.run(run())


def test_close_cancels_generation_and_rejects_future_renders(demo, tmp_path):
    async def run():
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def handle(request):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async with asyncio.timeout(3), httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            voice = DemoVoice(demo, tmp_path, client=client)
            rendering = asyncio.create_task(voice.render(Action('greeting'), {}))
            await entered.wait()
            await voice.close()
            assert cancelled.is_set()
            with pytest.raises(asyncio.CancelledError):
                await rendering
            assert not client.is_closed
            assert not [path for path in tmp_path.rglob('*') if path.is_file()]
            with pytest.raises(AudioBankError, match='closed'):
                await voice.render(Action('greeting'), {})
            await voice.close()

    asyncio.run(run())


def test_owned_client_closes_after_pending_producers_exit(demo, tmp_path, monkeypatch):
    async def run():
        entered, cancelled = asyncio.Event(), asyncio.Event()
        close_calls = []

        async def handle(request):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        class OwnedClient(httpx.AsyncClient):
            async def aclose(self):
                assert cancelled.is_set()
                close_calls.append(True)
                await super().aclose()

        client = OwnedClient(transport=httpx.MockTransport(handle))
        monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: client)
        async with asyncio.timeout(3):
            voice = DemoVoice(demo, tmp_path)
            rendering = asyncio.create_task(voice.render(Action('greeting'), {}))
            await entered.wait()
            await voice.close()
            with pytest.raises(asyncio.CancelledError):
                await rendering
            assert client.is_closed
            await voice.close()
            assert close_calls == [True]

    asyncio.run(run())


@pytest.mark.parametrize('cancel_first_close', [False, True], ids=['concurrent_close', 'cancel_then_retry'])
def test_all_close_callers_wait_for_shared_cleanup(demo, tmp_path, monkeypatch, cancel_first_close):
    async def run():
        entered = asyncio.Event()
        cleanup_started = asyncio.Event()
        cleanup_release = asyncio.Event()
        cleanup_finished = asyncio.Event()
        close_calls = []

        async def handle(request):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_started.set()
                await cleanup_release.wait()
                cleanup_finished.set()
                raise

        class OwnedClient(httpx.AsyncClient):
            async def aclose(self):
                assert cleanup_finished.is_set()
                close_calls.append(True)
                await super().aclose()

        client = OwnedClient(transport=httpx.MockTransport(handle))
        monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: client)
        async with asyncio.timeout(3):
            voice = DemoVoice(demo, tmp_path)
            rendering = asyncio.create_task(voice.render(Action('greeting'), {}))
            await entered.wait()
            first = asyncio.create_task(voice.close())
            await cleanup_started.wait()
            if cancel_first_close:
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
            second = asyncio.create_task(voice.close())
            await asyncio.sleep(0)
            assert not second.done()
            assert not cleanup_finished.is_set()
            assert not client.is_closed
            with pytest.raises(AudioBankError, match='closed'):
                await voice.render(Action('greeting'), {})
            cleanup_release.set()
            await second
            if not cancel_first_close:
                await first
            with pytest.raises(asyncio.CancelledError):
                await rendering
            assert client.is_closed
            await voice.close()
            assert close_calls == [True]
            assert not [path for path in tmp_path.rglob('*') if path.is_file()]

    asyncio.run(run())


def test_xtts_api_name_and_cache_revision_have_distinct_identities(demo, tmp_path):
    settings = deepcopy(demo['demo'])
    settings.update(settings['darija_xtts'])
    settings['tts_provider'] = 'darija_xtts'
    revised = dict(settings, tts_cache_revision='other_api')
    other_api = dict(settings, tts_api_name='other_api')
    text = 'Offline identity fixture'
    paths = {speech_cache_path(choice, tmp_path, text)
             for choice in (settings, revised, other_api)}
    assert len(paths) == 3

    # Default API/no revision must keep previously verified cached clips usable.
    original_identity = ['darija_xtts', settings['tts_model'], settings['tts_voice'],
                         text, settings['tts_url'], settings['tts_temperature']]
    original_digest = hashlib.sha256(json.dumps(original_identity, ensure_ascii=False).encode()).hexdigest()
    assert speech_cache_path(settings, tmp_path, text).name == original_digest + '.wav'


@pytest.mark.parametrize('change', ['voice', 'text', 'model', 'endpoint'])
def test_changed_synthesis_identity_does_not_reuse_cache(demo, tmp_path, change):
    async def run():
        seen = []

        def handle(request):
            seen.append((str(request.url), json.loads(request.content)))
            return httpx.Response(200, content=bytes(1024))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            voice = DemoVoice(demo, tmp_path, client=client)
            await voice.render(Action('greeting'), {})
            changed = deepcopy(demo)
            if change == 'text':
                changed['demo']['responses']['greeting'] += ' مرحبا'
            else:
                key = {'voice': 'tts_voice', 'model': 'tts_model', 'endpoint': 'tts_url'}[change]
                changed['demo'][key] += 'different'
            other = DemoVoice(changed, tmp_path, client=client)
            await other.render(Action('greeting'), {})
            assert len(seen) == 2
            assert len(list(tmp_path.rglob('*.wav'))) == 2
            await voice.close()
            await other.close()

    asyncio.run(run())


def test_audio_is_validated_before_atomic_publication(demo, tmp_path, monkeypatch):
    published = []
    original_replace = os.replace

    def observe_replace(source, target):
        source, target = Path(source), Path(target)
        assert source.parent == target.parent
        assert source != target
        assert not target.exists()
        assert pcm16(source.read_bytes(), demo['runtime']) == bytes(1024)
        published.append(target)
        return original_replace(source, target)

    monkeypatch.setattr(os, 'replace', observe_replace)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=bytes(1024)),
        )) as client:
            voice = DemoVoice(demo, tmp_path, client=client)
            await voice.render(Action('greeting'), {})
            await voice.close()
        assert len(published) == 1
        assert [path for path in tmp_path.rglob('*') if path.is_file()] == published

    asyncio.run(run())


def test_failed_atomic_publication_removes_temporary_audio(demo, tmp_path, monkeypatch):
    def fail_replace(source, target):
        assert Path(source).is_file()
        raise OSError('Offline fixture publication failure')

    monkeypatch.setattr(os, 'replace', fail_replace)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=bytes(1024)),
        )) as client:
            voice = DemoVoice(demo, tmp_path, client=client)
            with pytest.raises((OSError, AudioBankError), match='publication|cache'):
                await voice.render(Action('greeting'), {})
            assert not [path for path in tmp_path.rglob('*') if path.is_file()]
            await voice.close()

    asyncio.run(run())


def test_corrupt_cache_is_replaced_only_after_successful_regeneration(demo, tmp_path):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def handle(request):
            nonlocal calls
            calls += 1
            if calls == 2:
                entered.set()
                await release.wait()
            return httpx.Response(200, content=bytes(1024))

        async with asyncio.timeout(3), httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            voice = DemoVoice(demo, tmp_path, client=client)
            await voice.render(Action('greeting'), {})
            path, = tmp_path.rglob('*.wav')
            path.write_bytes(b'corrupt previous cache entry')
            repairing = asyncio.create_task(voice.render(Action('greeting'), {}))
            await entered.wait()
            assert path.read_bytes() == b'corrupt previous cache entry'
            release.set()
            await repairing
            assert pcm16(path.read_bytes(), demo['runtime']) == bytes(1024)
            assert calls == 2
            await voice.close()

    asyncio.run(run())
