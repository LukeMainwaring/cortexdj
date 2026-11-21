"""Agent streaming endpoint.

Provides POST /agent/chat for the CortexDJ assistant, streaming responses
in Vercel AI SDK protocol format.

Conversation history lives server-side: the client sends only its newest message
(see ``frontend/components/chat.tsx``), the stored thread history is passed to the
run as ``message_history``, and the run's own messages are appended when it ends.

The user's turn is stored and committed *before* the run starts. ``on_complete``
fires only on success, so anything that ends the run early (a provider error, the
Stop button, a dropped connection) would otherwise discard a message the browser
is still showing, and the client no longer re-sends it.

The flip side: when a client regenerates that unanswered turn (the Retry on a failed
reply), the stored copy stands in for it rather than being stored a second time, so
the run and the history see the turn once. Only a regenerate counts: a run still in
flight also leaves history ending on the user's turn, so a duplicate *submit* is a
new turn, never a retry; reading it as one would give one turn two replies.

Two limits of that Retry, accepted rather than solved: a reply the server stored but the
browser never finished receiving (the connection dropped in the finish window) leaves
history ending on the reply, so the retried turn is stored and answered again; and a
retry re-runs the whole turn, so a tool with a side effect that ran before the run died
runs again (the dead run's tool calls are never stored). A true regenerate would delete
the stored messages after the user turn; add that when the frontend has a Regenerate
button on answered replies. Pydantic AI's ``manage_message_history='server'`` is worth
weighing then: it cuts the run's history at a resubmitted turn by UI message id, which
storage drops today (``_without_client_ids``), and leaves the stored rows to us.

Do not wrap the turn in a span of our own: ``streaming_response`` is lazy, so the
agent runs after the handler returns, and a span held open across that boundary ends
late, in another context, when the client disconnects. The agent name and thread id
are already on the ``invoke_agent`` span
(docs/adr/0003-run-metadata-not-a-custom-chat-turn-span.md).
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

from fastapi import APIRouter
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from pydantic_ai import DeferredToolResults
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelContextWindowExceededError,
    ModelNotFoundError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelRequestRejectedError,
    ModelUnavailableError,
    RunCancelled,
)
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.messages import _drop_compaction_parts as drop_compaction_parts  # private: what the run itself uses
from pydantic_ai.ui import UIEventStream
from pydantic_ai.ui._utils import without_ui_message_id  # private: the exact inverse of what sanitizing keeps
from pydantic_ai.ui.vercel_ai import VercelAIAdapter, VercelAIEventStream
from pydantic_ai.ui.vercel_ai.request_types import RequestData
from pydantic_ai.ui.vercel_ai.response_types import BaseChunk, ErrorChunk
from starlette.requests import Request
from starlette.responses import Response

from cortexdj.agents.deps import AgentDeps
from cortexdj.agents.registry import ASSISTANT, get_agent
from cortexdj.core.db import AsyncPostgresSessionDep
from cortexdj.features.chat import service as thread_service
from cortexdj.features.chat.serialization import extract_latest_user_text
from cortexdj.features.chat.title_generator import TitleGeneratorDep

logger = logging.getLogger(__name__)

agent_router = APIRouter(prefix="/agent", tags=["agent"])


# Provider failures the user can act on, each with what to tell them. Retrying is the right move
# for a busy provider and the wrong one for the rest, which the generic "Try again" implies.
# Authentication and not-found failures are server misconfiguration, so their messages point at
# the server, not the conversation. The first matching row wins: ModelRequestRejectedError is the
# parent of the context-window, authentication and not-found categories, so it stays last.
_USER_FACING_RUN_ERRORS: list[tuple[tuple[type[ModelAPIError], ...], str]] = [
    (
        (ModelRateLimitError, ModelUnavailableError),
        "The model provider is busy right now. Try again in a moment.",
    ),
    (
        (ModelContextWindowExceededError,),
        "This conversation is too long for the model. Start a new chat to keep going.",
    ),
    (
        (ModelQuotaExceededError,),
        "The model provider's quota or credits are used up. Retrying won't help until it is topped up.",
    ),
    (
        (ModelAuthenticationError,),
        "The model provider rejected the server's credentials or their access to this model. "
        "Retrying won't help until that is fixed.",
    ),
    (
        (ModelNotFoundError,),
        "The model provider couldn't find the configured model, or the server's provider URL is wrong. "
        "Retrying won't help until that is fixed.",
    ),
    (
        (ModelRequestRejectedError,),
        "The model provider refused this request. Retrying it unchanged won't help.",
    ),
]


def _describe_run_error(error: Exception) -> str | None:
    """The message to show the user for a run error, or None to leave the generic fallback."""
    for categories, message in _USER_FACING_RUN_ERRORS:
        if isinstance(error, categories):
            return message
    return None


class _ChatEventStream(VercelAIEventStream[AgentDeps, Any]):
    """Log the exception that ends a run early, and tell the user what it means when they can act on it.

    Without the log, a run that dies mid-stream (for example "Exceeded maximum output retries")
    leaves no server-side trace outside Logfire. (The ``invoke_agent`` span records the error itself.)

    A known provider failure (``_describe_run_error``) goes to the browser as a JSON ``detail``, the
    shape ``describeChatError`` in ``frontend/components/messages.tsx`` shows. Any other error keeps
    the parent's raw text, which the browser replaces with its generic fallback.
    """

    async def on_error(self, error: Exception) -> AsyncIterator[BaseChunk]:
        logger.exception(f"Agent run ended with an error mid-stream: {error!r}")
        message = _describe_run_error(error)
        # Do not drop the parent's chunks: they send the error to the browser, mark the
        # finish reason as an error, and close any tool call still streaming its input.
        # A `return` here (no yield) would give the client a normal finish and leave a
        # tool call spinning.
        async for chunk in super().on_error(error):
            if message is not None and isinstance(chunk, ErrorChunk):
                chunk = ErrorChunk(error_text=json.dumps({"detail": message}))
            yield chunk

    async def on_cancelled(self, cancelled: RunCancelled) -> AsyncIterator[BaseChunk]:
        # A first-party cancel is not an error: the parent sends an abort chunk, so the browser
        # sees a stopped turn, not a failed one. Logged at INFO for the same reason.
        logger.info(f"Agent run cancelled mid-stream: {cancelled!r}")
        async for chunk in super().on_cancelled(cancelled):
            yield chunk


class _ChatAdapter(VercelAIAdapter[AgentDeps, Any]):
    """The Vercel AI adapter with run errors logged and known provider failures explained."""

    def build_event_stream(self) -> UIEventStream[RequestData, BaseChunk, AgentDeps, Any]:
        return _ChatEventStream(
            self.run_input,
            accept=self.accept,
            sdk_version=self.sdk_version,
            server_message_id=self.server_message_id,
        )

    def sanitize_messages(
        self,
        messages: Sequence[ModelMessage],
        *,
        deferred_tool_results: DeferredToolResults | None = None,
    ) -> list[ModelMessage]:
        """The parent's sanitizing, with client compaction parts always dropped first.

        The route stores the turn this returns before the run, and stored history is trusted from
        then on, so a client-sent compaction part would later trim the real history from what the
        model sees. ``UIAdapter.sanitize_messages`` never looks at compaction parts; it is
        ``run_stream_native`` that drops them, and only when ``message_history`` is non-empty, so
        the first turn of a thread would keep one. Dropping them here, with the run's own filter,
        inside the method the run calls, makes the stored copy and the run's copy one result on
        every turn. Do not store the module-level ``sanitize_messages`` output instead: it strips
        the UI message id ``load_messages`` keeps, and warns on every turn.
        """
        return super().sanitize_messages(
            drop_compaction_parts(list(messages)), deferred_tool_results=deferred_tool_results
        )


def _user_prompts(message: ModelMessage) -> list[object]:
    if not isinstance(message, ModelRequest):
        return []
    return [part.content for part in message.parts if isinstance(part, UserPromptPart)]


def _is_resend_of_unanswered_turn(history: list[ModelMessage], incoming: list[ModelMessage]) -> bool:
    """True when the stored history ends with a user turn no run answered and ``incoming`` repeats it.

    History ends on a user request after a broken run, and also while a run is in flight (the
    turn is stored before the run, the reply only on success): the caller only asks on a regenerate.
    """
    if not history or len(incoming) != 1:
        return False
    prompts = _user_prompts(incoming[0])
    return bool(prompts) and _user_prompts(history[-1]) == prompts


def _without_client_ids(messages: list[ModelMessage]) -> list[ModelMessage]:
    """``messages`` without the UI message id the client chose, so stored history has server ids only.

    Sanitizing keeps the id so a run can hand the client its own message back. Storage cannot: a
    turn can legitimately be stored twice (the module docstring's Retry after a reply the browser
    never received), and the thread would then read back with two messages under one id, which
    the frontend keys on. A client could also pick an id on purpose to collide with a stored one.
    """
    return [without_ui_message_id(message) for message in messages]


# Strong refs to in-flight title tasks; asyncio only weakly references a running
# task, so a discarded create_task(...) can be GC'd mid-run.
_title_tasks: set[asyncio.Task[None]] = set()


@agent_router.post("/chat")
async def stream_chat(
    request: Request,
    db: AsyncPostgresSessionDep,
    generate_thread_title: TitleGeneratorDep,
) -> Response:
    """Assistant streaming endpoint.

    One ``VercelAIAdapter`` built from the request parses the body, sanitizes the client's turn
    for storage, and runs the agent: what ``dispatch_request`` does, opened up. The run sanitizes
    the same cached messages again on the same instance with the same approvals, so the stored
    copy and the model's copy agree by construction; the second call is the price of not reaching
    into the adapter's caches.

    A second agent is a swap on this adapter, ``replace(adapter, agent=get_agent(name))``, once
    the thread is loaded and before anything reads ``adapter.messages``: the replacement rebuilds
    that cached property, so swapping later splits the stored copy from the run's. The name is
    read from stored state, never from client input or a model (the registry holds no selection
    logic).
    """
    try:
        adapter = await _ChatAdapter.from_request(request, agent=get_agent(ASSISTANT), sdk_version=7)
    except ValidationError as e:
        # FastAPI's own 422, in the `detail` shape every other route's has: `loc` from the body
        # down, and no `input`, which would echo the raw turn (the user's own text, and not always UTF-8,
        # which the JSON encoder cannot take) into the response and onto the request span.
        errors = [
            {**error, "loc": ("body", *error["loc"])} for error in e.errors(include_input=False, include_url=False)
        ]
        raise RequestValidationError(errors) from e
    thread_id = adapter.run_input.id

    history = await thread_service.get_or_create_thread_history(db, thread_id)
    # Client messages are untrusted: sanitizing drops injected system prompts, unfetchable file
    # references, compaction parts, and unresolved trailing tool calls (the run resolves them
    # against the same approvals, so pass them). It keeps the client's UI message id; the stored
    # copy must not, see `_without_client_ids`.
    # No tool here is `requires_approval`. Adding one leaves one more case to settle: the approval
    # turn echoes the assistant message the previous run already stored, and the adapter appends
    # it to the history as-is, so the run and the stored copy would both carry the tool call twice.
    incoming = adapter.sanitize_messages(adapter.messages, deferred_tool_results=adapter.deferred_tool_results)
    user_query = extract_latest_user_text(adapter.run_input.messages)

    # Durable before the run — see the module docstring. The session commits again at
    # request end (core/db.py), which is what persists the agent's own messages.
    if adapter.run_input.trigger == "regenerate-message" and _is_resend_of_unanswered_turn(history, incoming):
        # The stored copy is the turn: run without it in the history and do not store it again.
        history = history[:-1]
    else:
        await thread_service.append_messages(db, thread_id, _without_client_ids(incoming))
    await db.commit()

    async def on_complete(result):  # type: ignore[no-untyped-def]
        # The adapter folds the client's turn into `message_history`, so `new_messages()`
        # is exactly what this run produced; the user's turn is already stored above.
        await thread_service.append_messages(db, thread_id, result.new_messages())
        thread = await thread_service.touch_thread(db, thread_id)

        if thread and thread.title is None and result.output:
            # Independent task, NOT BackgroundTasks: those run before this request
            # commits, and the touch above row-locks the thread that the title
            # UPDATE also needs — BackgroundTasks would deadlock.
            # The title is seeded from what the user read: for a structured output
            # that is the response's text, not the object's repr.
            assistant_response = result.output if isinstance(result.output, str) else result.response.text
            title_task = asyncio.create_task(
                generate_thread_title(
                    thread_id=thread_id,
                    user_message=user_query,
                    assistant_response=assistant_response or str(result.output),
                )
            )
            _title_tasks.add(title_task)
            title_task.add_done_callback(_title_tasks.discard)

    return adapter.streaming_response(
        adapter.run_stream(
            deps=AgentDeps(db=db),
            message_history=history,
            conversation_id=thread_id,
            on_complete=on_complete,
        )
    )
