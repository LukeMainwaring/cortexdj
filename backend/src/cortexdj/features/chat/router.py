import logging

from fastapi import APIRouter, HTTPException

from cortexdj.core.db import AsyncPostgresSessionDep
from cortexdj.features.chat import service as thread_service
from cortexdj.features.chat.schema import (
    ThreadDeleteResponse,
    ThreadListResponse,
    ThreadMessagesResponse,
    ThreadRenameRequest,
    ThreadRenameResponse,
    ThreadSummary,
)

logger = logging.getLogger(__name__)

thread_router = APIRouter(prefix="/threads", tags=["threads"])


def _thread_not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Thread not found")


@thread_router.get("")
async def list_threads(db: AsyncPostgresSessionDep) -> ThreadListResponse:
    """List all threads."""
    threads = await thread_service.list_threads(db)
    return ThreadListResponse(threads=[ThreadSummary.model_validate(thread) for thread in threads])


@thread_router.get("/{thread_id}/messages")
async def get_thread_messages(db: AsyncPostgresSessionDep, thread_id: str) -> ThreadMessagesResponse:
    """Get all messages for a specific thread."""
    frontend_messages = await thread_service.get_thread_messages(db, thread_id)
    if frontend_messages is None:
        raise _thread_not_found()
    return ThreadMessagesResponse(thread_id=thread_id, messages=frontend_messages)


@thread_router.delete("/{thread_id}")
async def delete_thread(db: AsyncPostgresSessionDep, thread_id: str) -> ThreadDeleteResponse:
    """Delete a thread and all its messages."""
    if not await thread_service.delete_thread(db, thread_id):
        raise _thread_not_found()
    logger.info(f"Deleted thread {thread_id}")
    return ThreadDeleteResponse(message="Thread deleted successfully")


@thread_router.patch("/{thread_id}")
async def rename_thread(db: AsyncPostgresSessionDep, thread_id: str, body: ThreadRenameRequest) -> ThreadRenameResponse:
    """Rename a thread."""
    thread = await thread_service.rename_thread(db, thread_id, body.title)
    if thread is None:
        raise _thread_not_found()
    logger.info(f"Renamed thread {thread_id}")
    return ThreadRenameResponse.model_validate(thread)
