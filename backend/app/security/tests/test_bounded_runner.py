import asyncio
import threading
import time

import pytest

from app.security.bounded_runner import BoundedDaemonRunner


def test_runner_never_starts_more_than_four_expired_workers() -> None:
    runner = BoundedDaemonRunner(max_workers=4)
    release = threading.Event()
    for _ in range(4):
        with pytest.raises(TimeoutError):
            runner.run(lambda: release.wait(), timeout=0.01)
    with pytest.raises(TimeoutError):
        runner.run(lambda: None, timeout=0.01)
    assert runner.max_active == 4
    release.set()


@pytest.mark.asyncio
async def test_async_wait_times_out_without_blocking_event_loop() -> None:
    runner = BoundedDaemonRunner(max_workers=1)
    release = threading.Event()
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while ticks < 3:
            await asyncio.sleep(0)
            ticks += 1

    ticker_task = asyncio.create_task(ticker())
    with pytest.raises(TimeoutError):
        await runner.arun(lambda: release.wait(), timeout=0.01)
    await ticker_task
    assert ticks == 3
    release.set()
