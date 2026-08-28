"""Agent streaming endpoint.

Provides POST /agent/chat for the CortexDJ brain assistant,
streaming responses in Vercel AI SDK protocol format.

The database is the authoritative conversation history: the browser sends only
its newest turn (see ``frontend/components/chat.tsx``), the stored thread history
is handed to the run as ``message_history``, and only the messages the run itself
produced are appended when it ends.

The user's turn is stored and committed *before* the run starts. ``on_complete``
fires only on success, so anything that ends the run early — a provider error, the
Stop button, a dropped connection — would otherwise discard a message the browser
is still showing and no longer re-sends.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic_ai.messages import sanitize_messages
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from starlette.requests import Request
from starlette.responses import Response

from cortexdj.agents.brain_agent import brain_agent
from cortexdj.agents.deps import AgentDeps
from cortexdj.core.detached_tasks import spawn_detached_task
from cortexdj.dependencies.db import AsyncPostgresSessionDep
from cortexdj.dependencies.eeg_model import EEGModelDep
from cortexdj.models.message import Message
from cortexdj.models.thread import Thread
from cortexdj.schemas.agent_type import AgentType
from cortexdj.services.spotify import get_spotify_client, get_user_spotify_client
from cortexdj.services.title_generator import generate_thread_title
from cortexdj.utils.message_serialization import (
    deserialize_messages,
    extract_latest_user_text,
    prepare_messages_for_storage,
)

logger = logging.getLogger(__name__)

agent_router = APIRouter(prefix="/agent", tags=["agent"])


@agent_router.post("/chat")
async def stream_chat(
    request: Request,
    db: AsyncPostgresSessionDep,
    eeg_model: EEGModelDep,
) -> Response:
    """Brain assistant streaming endpoint.

    Uses VercelAIAdapter to handle parsing, agent execution, and streaming
    in Vercel AI SDK protocol format.
    """
    run_input = VercelAIAdapter.build_run_input(await request.body())
    thread_id = run_input.id
    agent_type = AgentType.CHAT.value

    history = deserialize_messages(await Message.get_history(db, thread_id, agent_type))
    # Sanitize exactly as the adapter does before the run, so what we store is what the
    # model was given. Browser messages are untrusted: this drops injected system prompts
    # and unfetchable file references. `strip_compaction_parts` mirrors what the adapter
    # does whenever `message_history` is passed — a client-supplied compaction boundary
    # would otherwise hide the trusted server-side history from the model.
    incoming = sanitize_messages(VercelAIAdapter.load_messages(run_input.messages), strip_compaction_parts=True)
    user_query = extract_latest_user_text(run_input.messages)

    # The thread row has to exist before any message row: messages carry a composite
    # foreign key onto it. Durable before the run — see the module docstring. The session
    # commits again at request end (dependencies/db.py), which persists the run's messages.
    thread_schema = await Thread.get_or_create(db, thread_id, agent_type)
    await Message.append_messages(db, thread_id, agent_type, prepare_messages_for_storage(incoming))
    await db.commit()

    spotify_client = await get_user_spotify_client(db) or get_spotify_client()

    deps = AgentDeps(
        db=db,
        eeg_model=eeg_model,
        spotify_client=spotify_client,
        thread_id=thread_id,
        brain_context=thread_schema.brain_context,
    )

    async def on_complete(result):  # type: ignore[no-untyped-def]
        # The adapter folds the browser's turn into `message_history`, so `new_messages()`
        # is exactly what this run produced; the user's turn is already stored above.
        await Message.append_messages(db, thread_id, agent_type, prepare_messages_for_storage(result.new_messages()))

        thread = await Thread.get(db, thread_id, agent_type)
        if thread:
            thread.updated_at = datetime.now(timezone.utc)
            await db.flush()

        if thread and thread.title is None and result.output:
            # Detached task, NOT BackgroundTasks: those run before this request commits,
            # and the updated_at flush above row-locks the thread that the title UPDATE
            # also needs — BackgroundTasks would deadlock. Keeps title generation off the
            # response's critical path.
            spawn_detached_task(
                generate_thread_title(
                    thread_id=thread_id,
                    agent_type=agent_type,
                    user_message=user_query,
                    assistant_response=str(result.output),
                ),
                name=f"generate-thread-title:{thread_id}",
            )

    return await VercelAIAdapter.dispatch_request(
        request,
        agent=brain_agent,
        deps=deps,
        message_history=history,
        conversation_id=thread_id,
        on_complete=on_complete,
        sdk_version=7,
    )
