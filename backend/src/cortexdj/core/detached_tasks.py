"""Strong references for fire-and-forget asyncio tasks.

The event loop only holds a *weak* reference to a running task, so a bare
``asyncio.create_task(...)`` whose handle nobody keeps can be garbage-collected
mid-run — the work silently stops partway. ``spawn_detached_task`` keeps the
handle until the task finishes, drops it in a done callback, and logs whatever
escaped it. Without that last step an unobserved failure only ever surfaces as
asyncio's "Task exception was never retrieved" at interpreter shutdown, detached
from the request that started it.

Do not use this for work the response depends on: nothing awaits these tasks, and
a task still running at shutdown is cancelled. Today's only caller is thread-title
generation in ``routers/agent.py``, which is deliberately off the critical path.
"""

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)

_pending: set[asyncio.Task[None]] = set()


def spawn_detached_task(coro: Coroutine[Any, Any, None], *, name: str) -> asyncio.Task[None]:
    """Start ``coro`` as a background task that is retained until it completes."""
    task = asyncio.create_task(coro, name=name)
    _pending.add(task)
    task.add_done_callback(_release_and_report)
    return task


def pending_detached_tasks() -> frozenset[asyncio.Task[None]]:
    """Snapshot of the tasks currently held; the retention seam tests observe."""
    return frozenset(_pending)


def _release_and_report(task: asyncio.Task[None]) -> None:
    _pending.discard(task)
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.error(f"Detached task {task.get_name()} failed: {error!r}", exc_info=error)
