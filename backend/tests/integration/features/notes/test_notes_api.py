"""Notes CRUD through the HTTP interface: create/get/list/404."""

from datetime import UTC, datetime, timedelta

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.factories import create_note


async def test_create_then_get_note(client: AsyncClient) -> None:
    created = await client.post("/api/notes", json={"title": "Groceries", "content": "milk"})
    assert created.status_code == 201
    note_id = created.json()["id"]

    fetched = await client.get(f"/api/notes/{note_id}")
    assert fetched.status_code == 200
    assert fetched.json()["title"] == "Groceries"
    assert fetched.json()["content"] == "milk"


async def test_get_unknown_note_returns_404(client: AsyncClient) -> None:
    resp = await client.get("/api/notes/does-not-exist")
    assert resp.status_code == 404


async def test_list_notes_newest_first(client: AsyncClient, db_session: AsyncSession) -> None:
    # Seed with explicit timestamps: within one transaction now() is constant, so
    # API-created rows would share created_at and order non-deterministically.
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    await create_note(db_session, title="older", content="a", created_at=t0)
    await create_note(db_session, title="newer", content="b", created_at=t0 + timedelta(hours=1))

    resp = await client.get("/api/notes")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert [n["title"] for n in body["notes"]] == ["newer", "older"]
