"""Deterministic pacing checks; no socket, providers, or wall-clock sleeps."""

import asyncio

import pytest

from bench.voice_transport import FramePacer


class FakeClock:
    def __init__(self, lateness=0.0):
        self.now = 0.0
        self.lateness = lateness
        self.waits = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds + self.lateness


def test_first_frame_is_immediate():
    clock = FakeClock()
    pacer = FramePacer(.032, clock=clock, sleeper=clock.sleep)
    assert asyncio.run(pacer.wait_next())
    assert clock.now == 0 and clock.waits == []
    assert pacer.deadline == pytest.approx(.032)


def test_fifteen_ms_late_wakes_do_not_accumulate_drift():
    async def exercise():
        clock = FakeClock(lateness=.015)
        pacer = FramePacer(.032, clock=clock, sleeper=clock.sleep)
        sent = []
        for _ in range(125):
            assert await pacer.wait_next()
            sent.append(clock.now)
        return clock, pacer, sent

    clock, pacer, sent = asyncio.run(exercise())
    assert sent[1] == pytest.approx(.047)
    assert sent[-1] == pytest.approx(124 * .032 + .015)
    assert [(b - a) for a, b in zip(sent[1:], sent[2:])] == pytest.approx([.032] * 123)
    assert sum(clock.waits[1:]) / len(clock.waits[1:]) == pytest.approx(.017)
    assert pacer.resets == 0
    assert pacer.max_lag_seconds == pytest.approx(.015)


@pytest.mark.parametrize('stall_during_sleep', [False, True])
def test_long_stall_resets_without_catch_up_burst(stall_during_sleep):
    async def exercise():
        clock = FakeClock()
        pacer = FramePacer(.032, clock=clock, sleeper=clock.sleep)
        assert await pacer.wait_next()
        if stall_during_sleep:
            clock.lateness = .500
        else:
            clock.now += .500  # A blocked websocket send or scheduling pause.
        assert await pacer.wait_next()
        after_stall = clock.now
        clock.lateness = 0
        sent = [after_stall]
        for _ in range(3):
            assert await pacer.wait_next()
            sent.append(clock.now)
        return pacer, sent

    pacer, sent = asyncio.run(exercise())
    assert pacer.resets == 1
    assert [b - a for a, b in zip(sent, sent[1:])] == pytest.approx([.032] * 3)


def test_stop_before_first_frame_does_not_sleep_or_advance():
    clock = FakeClock()
    pacer = FramePacer(.032, clock=clock, sleeper=clock.sleep)
    assert not asyncio.run(pacer.wait_next(lambda: True))
    assert pacer.deadline is None and not clock.waits


def test_stop_during_wait_does_not_admit_next_frame():
    async def exercise():
        clock = FakeClock()
        stopped = False

        async def stop_in_sleep(seconds):
            nonlocal stopped
            await clock.sleep(seconds)
            stopped = True

        pacer = FramePacer(.032, clock=clock, sleeper=stop_in_sleep)
        assert await pacer.wait_next(lambda: stopped)
        assert not await pacer.wait_next(lambda: stopped)
        return pacer

    assert asyncio.run(exercise()).deadline == pytest.approx(.032)


def test_cancellation_propagates_from_wait():
    async def exercise():
        clock = FakeClock()

        async def cancelled_sleep(seconds):
            raise asyncio.CancelledError()

        pacer = FramePacer(.032, clock=clock, sleeper=cancelled_sleep)
        assert await pacer.wait_next()
        with pytest.raises(asyncio.CancelledError):
            await pacer.wait_next()
        assert pacer.deadline == pytest.approx(.032)

    asyncio.run(exercise())


def test_early_wake_is_waited_out():
    async def exercise():
        clock = FakeClock()

        async def early_once(seconds):
            clock.waits.append(seconds)
            clock.now += seconds / 2 if len(clock.waits) == 1 else seconds

        pacer = FramePacer(.032, clock=clock, sleeper=early_once)
        assert await pacer.wait_next()
        assert await pacer.wait_next()
        return clock

    clock = asyncio.run(exercise())
    assert clock.now == pytest.approx(.032)
    assert clock.waits == pytest.approx([.032, .016])


@pytest.mark.parametrize('interval', [0, -.01, float('nan'), float('inf')])
def test_invalid_interval_rejected(interval):
    with pytest.raises(ValueError):
        FramePacer(interval)
