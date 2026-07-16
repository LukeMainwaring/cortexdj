"""Route-level tests for the thread endpoints, asserted through HTTP."""

from httpx import AsyncClient
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.models.message import Message
from cortexdj.schemas.agent_type import AgentType
from cortexdj.utils.message_serialization import prepare_messages_for_storage
from tests.factories import create_thread

_CHAT = AgentType.CHAT.value


def _history_payload() -> list[dict[str, object]]:
    return prepare_messages_for_storage(
        [
            ModelRequest(parts=[UserPromptPart(content="How was my last session?")]),
            ModelResponse(parts=[TextPart(content="It was mostly relaxed.")]),
        ]
    )


async def test_list_threads_empty(client: AsyncClient) -> None:
    response = await client.get("/api/threads")
    assert response.status_code == 200
    assert response.json()["threads"] == []


async def test_thread_messages_round_trip(client: AsyncClient, db_session: AsyncSession) -> None:
    thread = await create_thread(db_session)
    await Message.save_history(db_session, thread.thread_id, _CHAT, _history_payload())

    response = await client.get(f"/api/threads/{thread.thread_id}/messages")
    assert response.status_code == 200
    body = response.json()
    assert body["thread_id"] == thread.thread_id
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant"]


async def test_messages_for_unknown_thread_returns_404(client: AsyncClient) -> None:
    response = await client.get("/api/threads/no-such-thread/messages")
    assert response.status_code == 404


async def test_rename_thread(client: AsyncClient, db_session: AsyncSession) -> None:
    thread = await create_thread(db_session)

    response = await client.patch(f"/api/threads/{thread.thread_id}", json={"title": "Focus session recap"})
    assert response.status_code == 200
    assert response.json()["title"] == "Focus session recap"

    listed = (await client.get("/api/threads")).json()["threads"]
    assert listed[0]["title"] == "Focus session recap"


async def test_rename_unknown_thread_returns_404(client: AsyncClient) -> None:
    # Pins the ThreadNotFound(HTTPException) convention end-to-end.
    response = await client.patch("/api/threads/no-such-thread", json={"title": "x"})
    assert response.status_code == 404


async def test_delete_thread_cascades_messages(client: AsyncClient, db_session: AsyncSession) -> None:
    thread = await create_thread(db_session)
    await Message.save_history(db_session, thread.thread_id, _CHAT, _history_payload())

    response = await client.delete(f"/api/threads/{thread.thread_id}")
    assert response.status_code == 200

    assert (await client.get("/api/threads")).json()["threads"] == []
    # FK ondelete=CASCADE — exercised against the real schema.
    assert await Message.get_history(db_session, thread.thread_id, _CHAT) == []


async def test_delete_unknown_thread_returns_404(client: AsyncClient) -> None:
    response = await client.delete("/api/threads/no-such-thread")
    assert response.status_code == 404
