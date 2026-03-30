"""Thread service layer for business logic."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.models.message import Message
from cortexdj.models.thread import Thread, ThreadNotFound
from cortexdj.utils.message_serialization import dump_messages_for_frontend


async def list_threads(db: AsyncSession, agent_type: str) -> list[Thread]:
    """List all threads filtered by agent type."""
    return await Thread.list_all(db, agent_type)


async def get_thread_messages(db: AsyncSession, thread_id: str, agent_type: str) -> tuple[Thread, list[dict[str, Any]]]:
    """Get a thread and all its messages."""
    thread = await Thread.get(db, thread_id, agent_type)
    if not thread:
        raise ThreadNotFound()
    raw = await Message.get_history(db, thread_id, agent_type)
    return thread, dump_messages_for_frontend(raw)


async def delete_thread(db: AsyncSession, thread_id: str, agent_type: str) -> None:
    """Delete a thread and its messages."""
    thread = await Thread.get(db, thread_id, agent_type)
    if not thread:
        raise ThreadNotFound()
    await Thread.delete_by_id(db, thread_id, agent_type)


async def rename_thread(db: AsyncSession, thread_id: str, agent_type: str, title: str) -> Thread:
    """Rename a thread."""
    thread = await Thread.get(db, thread_id, agent_type)
    if not thread:
        raise ThreadNotFound()
    thread.title = title
    await db.flush()
    return thread
