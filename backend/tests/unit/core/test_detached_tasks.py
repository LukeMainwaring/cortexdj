"""Retention for fire-and-forget tasks: held while pending, released when done, failures logged."""

import asyncio
import logging

import pytest

from cortexdj.core.detached_tasks import pending_detached_tasks, spawn_detached_task

pytestmark = pytest.mark.anyio


async def test_task_is_retained_while_pending_and_released_when_done() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def work() -> None:
        started.set()
        await release.wait()

    task = spawn_detached_task(work(), name="retained-task")
    await started.wait()

    assert task in pending_detached_tasks(), "a running task must be strongly referenced, or it can be GC'd"

    release.set()
    await task

    assert task not in pending_detached_tasks(), "the done callback must drop the reference"


async def test_task_failure_is_logged_not_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    async def boom() -> None:
        raise RuntimeError("title service exploded")

    with caplog.at_level(logging.ERROR, logger="cortexdj.core.detached_tasks"):
        task = spawn_detached_task(boom(), name="failing-task")
        with pytest.raises(RuntimeError):
            await task
        # The done callback runs on the next loop iteration, after `await task` resumes.
        await asyncio.sleep(0)

    assert task not in pending_detached_tasks()
    assert any("failing-task" in record.message for record in caplog.records)
    assert any(record.exc_info is not None for record in caplog.records)


async def test_cancelled_task_is_released_without_logging(caplog: pytest.LogCaptureFixture) -> None:
    async def forever() -> None:
        await asyncio.Event().wait()

    with caplog.at_level(logging.ERROR, logger="cortexdj.core.detached_tasks"):
        task = spawn_detached_task(forever(), name="cancelled-task")
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)

    assert task not in pending_detached_tasks()
    assert caplog.records == [], "shutdown cancellation is expected, not an error to report"
