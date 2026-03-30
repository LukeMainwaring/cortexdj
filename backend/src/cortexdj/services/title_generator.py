"""Title generation service for chat threads."""

import logging

from openai import AsyncOpenAI

from cortexdj.dependencies.db import get_async_sqlalchemy_session
from cortexdj.models.thread import Thread

logger = logging.getLogger(__name__)

TITLE_MODEL = "gpt-4o-mini"

TITLE_INSTRUCTIONS = (
    "Generate a short, descriptive title (3-6 words) for this conversation. "
    "Return ONLY the title text, no quotes or punctuation at the end."
)

MAX_FALLBACK_LENGTH = 40


def _create_fallback_title(user_message: str) -> str:
    """Create a fallback title from the user message when LLM fails."""
    title = user_message.strip()
    if len(title) <= MAX_FALLBACK_LENGTH:
        return title
    truncated = title[:MAX_FALLBACK_LENGTH].rsplit(" ", 1)[0]
    return f"{truncated}..."


async def generate_thread_title(
    thread_id: str,
    agent_type: str,
    user_message: str,
    assistant_response: str,
) -> None:
    """Background task to generate and save thread title.

    Creates its own database session since this runs independently
    of the original request context.
    """
    try:
        client = AsyncOpenAI()
        response = await client.responses.create(
            model=TITLE_MODEL,
            instructions=TITLE_INSTRUCTIONS,
            input=f"User: {user_message[:500]}\n\nAssistant: {assistant_response[:500]}",
            max_output_tokens=20,
            store=False,
        )

        title = response.output_text
        if title:
            title = title.strip().strip('"').strip("'")
            async with get_async_sqlalchemy_session() as db:
                await Thread.update_title(db, thread_id, agent_type, title)
                await db.commit()
            logger.info(f"Generated title for thread {thread_id}: {title}")

    except Exception:
        logger.exception(f"Failed to generate title for thread {thread_id}")
        fallback = _create_fallback_title(user_message)
        try:
            async with get_async_sqlalchemy_session() as db:
                await Thread.update_title(db, thread_id, agent_type, fallback)
                await db.commit()
            logger.info(f"Used fallback title for thread {thread_id}: {fallback}")
        except Exception:
            logger.exception(f"Failed to save fallback title for thread {thread_id}")
