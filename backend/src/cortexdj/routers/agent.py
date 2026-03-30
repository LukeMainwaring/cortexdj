"""Agent streaming endpoint.

Provides POST /agent/chat for the CortexDJ brain assistant,
streaming responses in Vercel AI SDK protocol format.
"""

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from starlette.requests import Request
from starlette.responses import Response

from cortexdj.agents.brain_agent import brain_agent
from cortexdj.agents.deps import AgentDeps
from cortexdj.dependencies.db import AsyncPostgresSessionDep
from cortexdj.models.message import Message
from cortexdj.models.thread import Thread
from cortexdj.schemas.agent_type import AgentType
from cortexdj.schemas.thread import BrainContext
from cortexdj.services.spotify import get_spotify_client
from cortexdj.services.title_generator import generate_thread_title
from cortexdj.utils.message_serialization import extract_latest_user_text, prepare_messages_for_storage

logger = logging.getLogger(__name__)

agent_router = APIRouter(prefix="/agent", tags=["agent"])


def _get_eeg_model(request: Request):  # type: ignore[no-untyped-def]
    """Get EEG model from app state (loaded in lifespan)."""
    return getattr(request.app.state, "eeg_model", None)


@agent_router.post("/chat")
async def stream_chat(
    request: Request,
    db: AsyncPostgresSessionDep,
) -> Response:
    """Brain assistant streaming endpoint.

    Uses VercelAIAdapter to handle parsing, agent execution, and streaming
    in Vercel AI SDK protocol format.
    """
    body = await request.body()
    run_input = VercelAIAdapter.build_run_input(body)
    thread_id = run_input.id

    eeg_model = _get_eeg_model(request)
    spotify_client = get_spotify_client()

    # Load existing brain context for this thread
    thread = await Thread.get(db, thread_id, AgentType.CHAT.value)
    existing_context = BrainContext.model_validate(thread.brain_context) if thread and thread.brain_context else None

    deps = AgentDeps(
        db=db,
        eeg_model=eeg_model,
        spotify_client=spotify_client,
        thread_id=thread_id,
        brain_context=existing_context,
    )

    user_query = extract_latest_user_text(run_input.messages)

    async def on_complete(result):  # type: ignore[no-untyped-def]
        all_msgs = prepare_messages_for_storage(result.all_messages())
        await Thread.get_or_create(db, thread_id, AgentType.CHAT.value)
        await Message.save_history(db, thread_id, AgentType.CHAT.value, all_msgs)

        thread = await Thread.get(db, thread_id, AgentType.CHAT.value)
        if thread:
            thread.updated_at = datetime.now(timezone.utc)
            await db.flush()

        if thread and thread.title is None and result.output:
            asyncio.create_task(
                generate_thread_title(
                    thread_id=thread_id,
                    agent_type=AgentType.CHAT.value,
                    user_message=user_query,
                    assistant_response=str(result.output),
                )
            )

    return await VercelAIAdapter.dispatch_request(
        request,
        agent=brain_agent,
        deps=deps,
        on_complete=on_complete,
        sdk_version=6,
    )
