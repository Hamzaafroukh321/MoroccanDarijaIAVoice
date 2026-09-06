"""Router duration measurements keep submillisecond precision on Windows."""
import asyncio
import json
import time

import httpx
import pytest

import engine.router as router_module
from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.router import Router, RouterError
from engine.scoped_task import ScopedTaskState


class Clock:
    """Replace only the router binding; asyncio retains its real clock."""
    def __init__(self):
        self.reads = 0
        self.monotonic_reads = 0

    def monotonic(self):
        self.monotonic_reads += 1
        return 500.0

    def perf_counter(self):
        self.reads += 1
        return 500.0 + self.reads * .000125


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    config = demo_config(load_config(ROOT / 'configs/clinic.json'))
    config['router']['retries'] = 0
    clock = Clock()
    original_monotonic = time.monotonic
    monkeypatch.setattr(router_module, 'time', clock)
    assert time.monotonic is original_monotonic
    return config, clock


def answer(**changes):
    result = dict(intent='help', ops=[], confidence=.99, unclear=False,
        is_affirmation=False, is_negation=False, clarification=None, proposed_ops=[],
        resolves_clarification=None, discard_clarification=None, discard_request=None)
    result.update(changes)
    return result


def successful_reply():
    return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
        'message': {'content': json.dumps(answer())}}]})


def assert_duration(call):
    assert call['elapsed_clock'] == 'perf_counter'
    assert call['elapsed_ms'] == pytest.approx(.125, abs=.000001)
    assert 0 < call['elapsed_ms'] < 1


def test_exact_answer_retains_submillisecond_duration_without_provider_or_quota(setup, tmp_path):
    config, clock = setup
    task = ScopedTaskState(config)
    task.consume(answer(intent='ambiguous', clarification=dict(kind='ambiguous_value',
        slot='doctor', item_ids=[], coupled_slots=['time'])))

    async def run():
        def forbidden(*args, **kwargs):
            pytest.fail('Configured exact answer must not use the provider or quota')
        async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
            router = Router(config, tmp_path, api_key='', client=client)
            router.limiter.reserve = forbidden
            result = await router.route('doctor B', task.router_context())
            assert result.resolves_clarification == task.pending_clarification['id']
            assert result.ops[0].value == 'doctor_b'
            assert router.calls == [] and len(router.local_calls) == 1
            assert router.local_calls[0]['source'] == 'configured_exact_answer'
            assert_duration(router.local_calls[0])
    asyncio.run(run())
    assert clock.reads == 2 and clock.monotonic_reads == 0


@pytest.mark.parametrize('outcome', ['success', 'http_error', 'output_error', 'configuration_error'])
def test_provider_attempts_use_same_duration_clock_on_success_and_all_failure_categories(setup, tmp_path, outcome):
    config, clock = setup
    requests, reservations = [], []

    async def run():
        def reply(request):
            requests.append(request)
            if outcome == 'success':
                return successful_reply()
            if outcome == 'output_error':
                return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
                    'message': {'content': 'not valid json'}}]})
            if outcome == 'configuration_error':
                return httpx.Response(400, json={'error': {'type': 'invalid_request_error',
                    'param': 'response_format'}})
            return httpx.Response(503)
        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
            router = Router(config, tmp_path, api_key='offline-fixture', client=client)
            router.limiter.reserve = lambda: reservations.append(True)
            if outcome == 'success':
                assert (await router.route('please help', {'state': {}})).intent == 'help'
            else:
                with pytest.raises(RouterError):
                    await router.route('please help', {'state': {}})
            assert len(requests) == len(reservations) == len(router.calls) == 1
            assert router.local_calls == []
            assert router.calls[0]['ok'] is (outcome == 'success')
            assert_duration(router.calls[0])
    asyncio.run(run())
    assert clock.monotonic_reads == 0


def test_retry_durations_restart_at_each_attempt_instead_of_accumulating(setup, tmp_path):
    config, clock = setup
    config['router']['retries'] = 1
    requests, reservations = [], []

    async def run():
        def reply(request):
            requests.append(request)
            return httpx.Response(503) if len(requests) == 1 else successful_reply()
        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
            router = Router(config, tmp_path, api_key='offline-fixture', client=client)
            router.limiter.reserve = lambda: reservations.append(True)
            assert (await router.route('please help', {'state': {}})).intent == 'help'
            assert len(requests) == len(reservations) == len(router.calls) == 2
            assert [item['ok'] for item in router.calls] == [False, True]
            for item in router.calls:
                assert_duration(item)
    asyncio.run(run())
    assert clock.monotonic_reads == 0
