from typing import Any

from pydantic_ai.messages import ModelMessage
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.features.chat.model import Message, Thread
from cortexdj.features.chat.serialization import (
    deserialize_messages,
    dump_messages_for_frontend,
    prepare_messages_for_storage,
)


async def list_threads(db: AsyncSession) -> list[Thread]:
    return await Thread.list_all(db)


async def get_thread_messages(db: AsyncSession, thread_id: str) -> list[dict[str, Any]] | None:
    """The thread's messages in the frontend's shape, or None when there is no such thread."""
    if await Thread.get(db, thread_id) is None:
        return None
    return dump_messages_for_frontend(await Message.get_history(db, thread_id))


async def delete_thread(db: AsyncSession, thread_id: str) -> bool:
    return await Thread.delete_by_id(db, thread_id)


async def rename_thread(db: AsyncSession, thread_id: str, title: str) -> Thread | None:
    return await Thread.update_title(db, thread_id, title)


async def get_or_create_thread_history(db: AsyncSession, thread_id: str) -> list[ModelMessage]:
    """The thread's stored history as an agent run takes it, starting the thread on its first turn."""
    await Thread.get_or_create(db, thread_id)
    return deserialize_messages(await Message.get_history(db, thread_id))


async def append_messages(db: AsyncSession, thread_id: str, messages: list[ModelMessage]) -> None:
    """Append messages to the thread's history; the caller decides which messages are new."""
    await Message.append_messages(db, thread_id, prepare_messages_for_storage(messages))


async def touch_thread(db: AsyncSession, thread_id: str) -> Thread | None:
    return await Thread.touch(db, thread_id)
