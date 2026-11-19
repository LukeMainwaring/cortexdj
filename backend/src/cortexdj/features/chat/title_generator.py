import logging
from typing import Annotated, Protocol

from fastapi import Depends

from cortexdj.agents.registry import THREAD_TITLE, get_agent
from cortexdj.core.db import get_async_sqlalchemy_session
from cortexdj.features.chat import service as thread_service

logger = logging.getLogger(__name__)

MAX_FALLBACK_LENGTH = 40


def _create_fallback_title(user_message: str) -> str:
    title = user_message.strip()
    if len(title) <= MAX_FALLBACK_LENGTH:
        return title
    truncated = title[:MAX_FALLBACK_LENGTH].rsplit(" ", 1)[0]
    return f"{truncated}..."


async def generate_thread_title(
    thread_id: str,
    user_message: str,
    assistant_response: str,
) -> None:
    """Background task to generate and save a thread title.

    Creates its own database session since this runs independently of the
    original request context. Falls back to a truncated user message if the
    model call fails (e.g. no provider key configured).
    """
    try:
        result = await get_agent(THREAD_TITLE).run(
            f"User: {user_message[:500]}\n\nAssistant: {assistant_response[:500]}"
        )
        title = (result.output or "").strip().strip('"').strip("'")
        if title:
            async with get_async_sqlalchemy_session() as db:
                await thread_service.rename_thread(db, thread_id, title)
            logger.info(f"Generated title for thread {thread_id}: {title}")
            return

        raise ValueError("empty title from model")

    except Exception:
        logger.exception(f"Failed to generate title for thread {thread_id}")
        fallback = _create_fallback_title(user_message)
        try:
            async with get_async_sqlalchemy_session() as db:
                await thread_service.rename_thread(db, thread_id, fallback)
            logger.info(f"Used fallback title for thread {thread_id}: {fallback}")
        except Exception:
            logger.exception(f"Failed to save fallback title for thread {thread_id}")


class TitleGenerator(Protocol):
    async def __call__(self, *, thread_id: str, user_message: str, assistant_response: str) -> None: ...


def get_title_generator() -> TitleGenerator:
    """Provide the title generator the chat stream schedules after a thread's first reply.

    A dependency rather than a direct call because the generator opens its own database
    session and runs a model, outside the request: this is the seam a caller that must
    not do either (the integration tests) overrides through ``app.dependency_overrides``.
    Do not patch ``generate_thread_title`` in tests; override this instead
    (docs/adr/0001-test-tiers-and-fakes.md).
    """
    return generate_thread_title


TitleGeneratorDep = Annotated[TitleGenerator, Depends(get_title_generator)]
