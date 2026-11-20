"""A conversation's brain context: the table, its endpoint, the tool that sets it, and the instruction that reads it."""

import pytest
from httpx import AsyncClient
from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.messages import InstructionPart, ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.agents.registry import ASSISTANT, get_agent
from cortexdj.features.chat.model import Thread
from cortexdj.features.sessions import service as session_service
from cortexdj.features.sessions import tools as session_tools
from cortexdj.features.sessions.model import ThreadBrainContext
from cortexdj.features.sessions.schema import BrainContext
from tests.factories import create_brain_context, create_thread, make_deps, make_run_context


async def test_merge_creates_the_row_then_overwrites_only_the_fields_set(db_session: AsyncSession) -> None:
    await create_thread(db_session, "t-ctx")

    first = await session_service.merge_brain_context(
        db_session, "t-ctx", BrainContext(latest_session_id="sess-1", dominant_mood="relaxed")
    )
    assert first == BrainContext(latest_session_id="sess-1", dominant_mood="relaxed")

    merged = await session_service.merge_brain_context(db_session, "t-ctx", BrainContext(dominant_mood="excited"))
    assert merged.dominant_mood == "excited"
    assert merged.latest_session_id == "sess-1", "fields not set on the update must survive the merge"


async def test_a_context_for_a_missing_thread_fails_and_leaves_the_session_usable(db_session: AsyncSession) -> None:
    with pytest.raises(IntegrityError):
        await session_service.merge_brain_context(db_session, "t-missing", BrainContext(dominant_mood="calm"))

    # The run's other tools share this session, so the failed insert must not poison it.
    assert await session_service.get_brain_context(db_session, "t-missing") is None


async def test_endpoint_returns_the_threads_context(client: AsyncClient, db_session: AsyncSession) -> None:
    await create_thread(db_session, "t-ctx")
    await create_brain_context(db_session, "t-ctx", dominant_mood="calm", avg_arousal=0.3)

    resp = await client.get("/api/brain-context/t-ctx")

    assert resp.status_code == 200
    assert resp.json() == {
        "latest_session_id": None,
        "dominant_mood": "calm",
        "avg_arousal": 0.3,
        "avg_valence": None,
    }


async def test_endpoint_returns_null_when_no_context_is_set(client: AsyncClient, db_session: AsyncSession) -> None:
    await create_thread(db_session, "t-ctx")

    resp = await client.get("/api/brain-context/t-ctx")

    assert resp.status_code == 200
    assert resp.json() is None


async def test_deleting_the_thread_deletes_its_context(client: AsyncClient, db_session: AsyncSession) -> None:
    await create_thread(db_session, "t-ctx")
    await create_brain_context(db_session, "t-ctx", dominant_mood="calm")

    deleted = await client.delete("/api/threads/t-ctx")
    assert deleted.status_code == 200

    # FK ondelete=CASCADE removed the row in the database; expire the identity map so the read goes there.
    db_session.expire_all()
    assert await ThreadBrainContext.get(db_session, "t-ctx") is None
    assert await Thread.get(db_session, "t-ctx") is None


async def test_set_brain_context_tool_stores_the_conversations_context(db_session: AsyncSession) -> None:
    await create_thread(db_session, "t-ctx")
    ctx = make_run_context(make_deps(db_session), conversation_id="t-ctx")

    result = await session_tools.set_brain_context(ctx, session_id="sess-1", dominant_mood="relaxed")

    assert "relaxed" in result
    assert await session_service.get_brain_context(db_session, "t-ctx") == BrainContext(
        latest_session_id="sess-1", dominant_mood="relaxed"
    )


async def test_set_brain_context_tool_fails_outside_a_conversation(db_session: AsyncSession) -> None:
    ctx = make_run_context(make_deps(db_session))

    with pytest.raises(ToolFailed, match="thread ID"):
        await session_tools.set_brain_context(ctx, dominant_mood="relaxed")


async def _instructions_seen(db_session: AsyncSession, conversation_id: str) -> str:
    seen: list[InstructionPart] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(info.model_request_parameters.instruction_parts or [])
        return ModelResponse(parts=[TextPart("ok")])

    agent = get_agent(ASSISTANT)
    with agent.override(model=FunctionModel(respond)):
        await agent.run("Hello", deps=make_deps(db_session), conversation_id=conversation_id)
    return "\n".join(p.content for p in seen)


async def test_each_request_carries_the_stored_context(db_session: AsyncSession) -> None:
    await create_thread(db_session, "t-ctx")
    await create_brain_context(db_session, "t-ctx", dominant_mood="relaxed")

    instructions = await _instructions_seen(db_session, "t-ctx")

    assert "## Active Brain Context" in instructions
    assert "- Dominant mood: relaxed" in instructions


async def test_a_request_without_stored_context_has_no_context_block(db_session: AsyncSession) -> None:
    await create_thread(db_session, "t-ctx")

    assert "Active Brain Context" not in await _instructions_seen(db_session, "t-ctx")


async def test_set_brain_context_tool_keeps_the_fields_a_call_does_not_name(db_session: AsyncSession) -> None:
    await create_thread(db_session, "t-ctx")
    ctx = make_run_context(make_deps(db_session), conversation_id="t-ctx")
    await session_tools.set_brain_context(ctx, session_id="sess-1", avg_arousal=0.3, avg_valence=0.7)

    await session_tools.set_brain_context(ctx, dominant_mood="relaxed")

    assert await session_service.get_brain_context(db_session, "t-ctx") == BrainContext(
        latest_session_id="sess-1", dominant_mood="relaxed", avg_arousal=0.3, avg_valence=0.7
    )
