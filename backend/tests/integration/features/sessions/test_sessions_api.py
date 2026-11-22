"""Route-level tests for the sessions endpoints, asserted through HTTP."""

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.factories import create_eeg_segment, create_session, create_session_track, create_track


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


async def test_segments_trajectory_is_counted_per_track(client: AsyncClient, db_session: AsyncSession) -> None:
    session = await create_session(db_session, duration_seconds=120.0)
    # The segments flip quadrant every window; the tracks they average into do not change.
    for i in range(30):
        await create_eeg_segment(
            db_session,
            session.id,
            segment_index=i,
            start_time=4.0 * i,
            end_time=4.0 * i + 4.0,
            dominant_state="excited" if i % 2 else "calm",
        )
    for order in range(2):
        track = await create_track(db_session)
        await create_session_track(db_session, session.id, track.id, track_order=order, dominant_state="excited")

    response = await client.get(f"/api/sessions/{session.id}/segments")
    assert response.status_code == 200
    trajectory = response.json()["trajectory_summary"]
    assert trajectory["transition_count"] == 0
    assert trajectory["dwell_fractions"]["excited"] == 1.0
    assert [p["start_time"] for p in trajectory["smoothed"]] == [0.0, 60.0]


async def test_segments_for_unknown_session_is_empty_not_404(client: AsyncClient) -> None:
    # Pins current behavior: the endpoint returns an empty list rather than 404.
    response = await client.get("/api/sessions/no-such-session/segments")
    assert response.status_code == 200
    assert response.json()["total"] == 0


async def test_list_sessions_enriched_counts_the_label_over_tracks(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    session = await create_session(db_session)
    # Most segments read stressed, but seven of ten tracks read relaxed.
    for i in range(3):
        await create_eeg_segment(db_session, session.id, segment_index=i, dominant_state="stressed")
    for order in range(10):
        track = await create_track(db_session)
        state = "relaxed" if order < 7 else "calm"
        await create_session_track(db_session, session.id, track.id, track_order=order, dominant_state=state)

    response = await client.get("/api/sessions/enriched")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    summary = body["sessions"][0]
    assert summary["display_index"] == 1
    assert summary["dominant_state"] == "relaxed"
    assert summary["label"] == "Mostly relaxed"
    assert summary["state_distribution"] == {"relaxed": 0.7, "calm": 0.3}
    assert summary["segment_count"] == 3
    assert summary["track_count"] == 10


async def test_list_sessions_enriched_breaks_a_tie_in_quadrant_order(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    session = await create_session(db_session)
    for order, state in enumerate(["stressed", "excited", "stressed", "excited"]):
        track = await create_track(db_session)
        await create_session_track(db_session, session.id, track.id, track_order=order, dominant_state=state)

    response = await client.get("/api/sessions/enriched")
    summary = response.json()["sessions"][0]
    assert summary["dominant_state"] == "excited"
    assert summary["label"] == "Mixed, leaning excited"


async def test_list_sessions_enriched_label_throughout(client: AsyncClient, db_session: AsyncSession) -> None:
    session = await create_session(db_session)
    for order in range(10):
        track = await create_track(db_session)
        state = "relaxed" if order < 9 else "calm"
        await create_session_track(db_session, session.id, track.id, track_order=order, dominant_state=state)

    response = await client.get("/api/sessions/enriched")
    summary = response.json()["sessions"][0]
    assert summary["label"] == "Relaxed throughout"
