"""The POST /agent/chat persistence contract: history lives server-side and survives a broken run.

The browser sends only its newest turn (see ``frontend/components/chat.tsx``), so
anything the server fails to store is gone for good — these cover both halves of that
bargain. The model double is a streaming ``FunctionModel``: the endpoint only ever
streams, so a plain (non-streaming) ``FunctionModel`` would never be called.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import AsyncClient
from pydantic_ai.messages import ModelMessage, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from cortexdj.agents.brain_agent import brain_agent
from cortexdj.core.detached_tasks import pending_detached_tasks

_CHAT_URL = "/api/agent/chat"


@pytest.fixture(autouse=True)
def _no_title_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Title generation calls a real model from a detached task on its own session — out of scope here."""

    async def _noop(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr("cortexdj.routers.agent.generate_thread_title", _noop)


def _submit(thread_id: str, text: str) -> dict[str, Any]:
    """One turn as the frontend sends it: the newest message only."""
    return {
        "trigger": "submit-message",
        "id": thread_id,
        "messages": [{"id": f"msg-{text}", "role": "user", "parts": [{"type": "text", "text": text}]}],
    }


async def _roles(client: AsyncClient, thread_id: str) -> list[str]:
    response = await client.get(f"/api/threads/{thread_id}/messages")
    assert response.status_code == 200
    return [message["role"] for message in response.json()["messages"]]


async def test_second_turn_is_answered_from_stored_history(client: AsyncClient) -> None:
    seen: list[list[ModelMessage]] = []

    async def reply(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        seen.append(messages)
        yield "ok"

    with brain_agent.override(model=FunctionModel(stream_function=reply)):
        first = await client.post(_CHAT_URL, json=_submit("t-turns", "first"))
        assert first.status_code == 200
        assert first.text  # the stream is consumed like a real client would
        second = await client.post(_CHAT_URL, json=_submit("t-turns", "second"))
        assert second.status_code == 200
        assert second.text

    # Turn 2 carried only its own message, yet the run saw turn 1 — read back from the database.
    second_turn_prompts = [p.content for m in seen[1] for p in m.parts if isinstance(p, UserPromptPart)]
    assert second_turn_prompts == ["first", "second"]

    # Each user turn stored once, each assistant result stored once.
    assert await _roles(client, "t-turns") == ["user", "assistant", "user", "assistant"]


async def test_broken_run_still_stores_the_user_turn(client: AsyncClient) -> None:
    async def die_mid_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield "partial"
        raise RuntimeError("provider is down")

    # The adapter reports a run error as an SSE error event, so the request itself still succeeds.
    with brain_agent.override(model=FunctionModel(stream_function=die_mid_stream)):
        response = await client.post(_CHAT_URL, json=_submit("t-broken", "keep me"))
        assert response.status_code == 200
        assert response.text

    # `on_complete` never fired, so no assistant result was persisted — but the user's
    # message survives, because the client will not re-send it.
    assert await _roles(client, "t-broken") == ["user"]


async def test_title_task_is_retained_until_it_finishes(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Title generation outlives the response, so something has to hold its task."""
    release = asyncio.Event()

    async def blocking_title(**kwargs: Any) -> None:
        await release.wait()

    monkeypatch.setattr("cortexdj.routers.agent.generate_thread_title", blocking_title)

    async def reply(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield "ok"

    with brain_agent.override(model=FunctionModel(stream_function=reply)):
        assert (await client.post(_CHAT_URL, json=_submit("t-title", "name this"))).status_code == 200

    title_tasks = [t for t in pending_detached_tasks() if t.get_name() == "generate-thread-title:t-title"]
    assert len(title_tasks) == 1, "a bare create_task() handle can be garbage-collected mid-run"

    release.set()
    await title_tasks[0]
    assert title_tasks[0] not in pending_detached_tasks()
