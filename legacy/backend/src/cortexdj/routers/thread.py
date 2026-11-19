import logging

from fastapi import APIRouter

from cortexdj.dependencies.db import AsyncPostgresSessionDep
from cortexdj.schemas.agent_type import AgentType
from cortexdj.schemas.thread import (
    BrainContext,
    ThreadDeleteResponse,
    ThreadListResponse,
    ThreadMessagesResponse,
    ThreadRenameRequest,
    ThreadRenameResponse,
    ThreadSummary,
)
from cortexdj.services import thread as thread_service

logger = logging.getLogger(__name__)

thread_router = APIRouter(prefix="/threads", tags=["threads"])


@thread_router.get("")
async def list_threads(db: AsyncPostgresSessionDep) -> ThreadListResponse:
    """List all threads."""
    threads = await thread_service.list_threads(db, AgentType.CHAT.value)
    return ThreadListResponse(
        threads=[
            ThreadSummary(
                id=thread.thread_id,
                thread_id=thread.thread_id,
                title=thread.title,
                brain_context=BrainContext.model_validate(thread.brain_context) if thread.brain_context else None,
                created_at=thread.created_at,
                updated_at=thread.updated_at,
            )
            for thread in threads
        ]
    )


@thread_router.get("/{thread_id}/messages")
async def get_thread_messages(db: AsyncPostgresSessionDep, thread_id: str) -> ThreadMessagesResponse:
    """Get all messages for a specific thread."""
    thread, frontend_messages = await thread_service.get_thread_messages(db, thread_id, AgentType.CHAT.value)
    brain_context = BrainContext.model_validate(thread.brain_context) if thread.brain_context else None
    return ThreadMessagesResponse(thread_id=thread_id, messages=frontend_messages, brain_context=brain_context)


@thread_router.delete("/{thread_id}")
async def delete_thread(db: AsyncPostgresSessionDep, thread_id: str) -> ThreadDeleteResponse:
    """Delete a thread and all its messages."""
    await thread_service.delete_thread(db, thread_id, AgentType.CHAT.value)
    logger.info(f"Deleted thread {thread_id}")
    return ThreadDeleteResponse(message="Thread deleted successfully")


@thread_router.patch("/{thread_id}")
async def rename_thread(db: AsyncPostgresSessionDep, thread_id: str, body: ThreadRenameRequest) -> ThreadRenameResponse:
    """Rename a thread."""
    thread = await thread_service.rename_thread(db, thread_id, AgentType.CHAT.value, body.title)
    logger.info(f"Renamed thread {thread_id}")
    return ThreadRenameResponse(thread_id=thread.thread_id, title=thread.title or body.title)
