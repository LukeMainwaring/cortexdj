"""Route-level tests for the sessions endpoints, asserted through HTTP."""

from datetime import UTC, datetime

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.factories import create_eeg_segment, create_session


async def test_list_sessions_empty(client: AsyncClient) -> None:
    response = await client.get("/api/sessions")
    assert response.status_code == 200
    body = response.json()
    assert body["sessions"] == []
    assert body["total"] == 0


async def test_list_sessions_pagination(client: AsyncClient, db_session: AsyncSession) -> None:
    for day in (1, 2, 3):
        await create_session(db_session, recorded_at=datetime(2024, 1, day, tzinfo=UTC))

    first_page = (await client.get("/api/sessions", params={"limit": 2, "offset": 0})).json()
    assert first_page["total"] == 3
    assert len(first_page["sessions"]) == 2
    # Newest first.
    assert first_page["sessions"][0]["recorded_at"].startswith("2024-01-03")

    second_page = (await client.get("/api/sessions", params={"limit": 2, "offset": 2})).json()
    assert second_page["total"] == 3
    assert len(second_page["sessions"]) == 1


async def test_get_session_by_id(client: AsyncClient, db_session: AsyncSession) -> None:
    session = await create_session(db_session, participant_id="P07")

    response = await client.get(f"/api/sessions/{session.id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == session.id
    assert body["participant_id"] == "P07"


async def test_get_unknown_session_returns_404(client: AsyncClient) -> None:
    response = await client.get("/api/sessions/no-such-session")
    assert response.status_code == 404


async def test_get_session_segments_ordered_with_trajectory(client: AsyncClient, db_session: AsyncSession) -> None:
    session = await create_session(db_session)
    # Insert out of order to prove ordering comes from segment_index.
    await create_eeg_segment(db_session, session.id, segment_index=1, start_time=4.0, end_time=8.0)
    await create_eeg_segment(db_session, session.id, segment_index=0, start_time=0.0, end_time=4.0)

    response = await client.get(f"/api/sessions/{session.id}/segments")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert [s["segment_index"] for s in body["segments"]] == [0, 1]
    assert body["trajectory_summary"] is not None


async def test_segments_for_unknown_session_is_empty_not_404(client: AsyncClient) -> None:
    # Pins current behavior: the endpoint returns an empty list rather than 404.
    response = await client.get("/api/sessions/no-such-session/segments")
    assert response.status_code == 200
    assert response.json()["total"] == 0


async def test_list_sessions_enriched(client: AsyncClient, db_session: AsyncSession) -> None:
    session = await create_session(db_session)
    for i in range(3):
        await create_eeg_segment(db_session, session.id, segment_index=i, dominant_state="relaxed")

    response = await client.get("/api/sessions/enriched")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    summary = body["sessions"][0]
    assert summary["display_index"] == 1
    assert summary["dominant_state"] == "relaxed"
    assert summary["label"] == "Relaxed throughout"
    assert summary["segment_count"] == 3
    assert summary["state_distribution"]["relaxed"] == 1.0
