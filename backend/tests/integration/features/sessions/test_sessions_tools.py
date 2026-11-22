"""Sessions agent tools against the real service and database, called the way the agent calls them."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.features.sessions import tools as session_tools
from cortexdj.features.sessions.eeg_model import override_eeg_model
from cortexdj.ml.model import EEGNetClassifier
from tests.factories import (
    create_eeg_segment,
    create_session,
    create_session_track,
    create_track,
    make_deps,
    make_run_context,
)


async def test_list_sessions_with_none_stored_says_how_to_seed(db_session: AsyncSession) -> None:
    result = await session_tools.list_sessions(make_run_context(make_deps(db_session)))

    assert result.startswith("No EEG sessions found")
    assert "seed-sessions" in result


async def test_list_sessions_labels_newest_first_with_the_id_for_tool_calls(db_session: AsyncSession) -> None:
    # Explicit timestamps: within one transaction now() is constant, and display indices follow created_at.
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    older = await create_session(db_session, created_at=t0)
    newer = await create_session(db_session, created_at=t0 + timedelta(hours=1))
    track = await create_track(db_session)
    await create_session_track(db_session, newer.id, track.id, dominant_state="relaxed")

    result = await session_tools.list_sessions(make_run_context(make_deps(db_session)), limit=1)

    assert "**2 EEG sessions** (showing 1)" in result
    assert "**Session 02** — Relaxed throughout · 1 tracks" in result
    assert f"<!-- id={newer.id} -->" in result
    assert older.id not in result


async def test_explain_brain_state_describes_the_dominant_state(db_session: AsyncSession) -> None:
    session = await create_session(db_session)
    await create_eeg_segment(db_session, session.id, segment_index=0, dominant_state="calm", arousal_score=0.2)
    await create_eeg_segment(db_session, session.id, segment_index=1, dominant_state="calm", arousal_score=0.4)

    result = await session_tools.explain_brain_state(make_run_context(make_deps(db_session)), session.id)

    assert "**Dominant State:** Calm" in result
    assert "**Average Arousal:** 0.30" in result
    assert '{"calm": 2}' in result


async def test_explain_brain_state_of_an_unknown_session_asks_for_a_listed_id(db_session: AsyncSession) -> None:
    with pytest.raises(ModelRetry, match="list_sessions"):
        await session_tools.explain_brain_state(make_run_context(make_deps(db_session)), "no-such-session")


async def test_explain_brain_state_of_a_session_without_segments_fails(db_session: AsyncSession) -> None:
    session = await create_session(db_session)

    with pytest.raises(ToolFailed, match="No EEG segments"):
        await session_tools.explain_brain_state(make_run_context(make_deps(db_session)), session.id)


async def test_compare_sessions_reports_the_deltas(db_session: AsyncSession) -> None:
    first = await create_session(db_session)
    second = await create_session(db_session)
    await create_eeg_segment(db_session, first.id, arousal_score=0.8, valence_score=0.6, dominant_state="excited")
    await create_eeg_segment(db_session, second.id, arousal_score=0.3, valence_score=0.6, dominant_state="relaxed")

    result = json.loads(
        await session_tools.compare_sessions(make_run_context(make_deps(db_session)), first.id, second.id)
    )

    assert result["comparison"]["arousal_delta"] == pytest.approx(0.5)
    assert result["comparison"]["valence_delta"] == pytest.approx(0.0)
    assert result["comparison"]["same_dominant_state"] is False


async def test_compare_sessions_with_an_unknown_session_asks_for_a_listed_id(db_session: AsyncSession) -> None:
    session = await create_session(db_session)

    with pytest.raises(ModelRetry, match="no-such-session"):
        await session_tools.compare_sessions(make_run_context(make_deps(db_session)), session.id, "no-such-session")


async def test_compare_sessions_fails_when_a_session_has_no_segments(db_session: AsyncSession) -> None:
    first = await create_session(db_session)
    second = await create_session(db_session)
    await create_eeg_segment(db_session, first.id)

    with pytest.raises(ToolFailed, match="no segments"):
        await session_tools.compare_sessions(make_run_context(make_deps(db_session)), first.id, second.id)


async def test_find_relaxing_tracks_with_no_data(db_session: AsyncSession) -> None:
    result = await session_tools.find_relaxing_tracks(make_run_context(make_deps(db_session)))

    assert result == {"tracks": [], "count": 0, "message": "No tracks with relaxation data found."}


async def test_find_relaxing_tracks_lists_relaxed_and_calm_tracks_once_each(db_session: AsyncSession) -> None:
    session = await create_session(db_session)
    relaxed = await create_track(db_session, title="Weightless", artist="Marconi Union")
    calm = await create_track(db_session, title="Gymnopedie", artist="Satie")
    loud = await create_track(db_session, title="Thunderstruck", artist="AC/DC")
    await create_session_track(db_session, session.id, relaxed.id, dominant_state="relaxed")
    await create_session_track(db_session, session.id, calm.id, track_order=1, dominant_state="calm")
    await create_session_track(db_session, session.id, calm.id, track_order=2, dominant_state="relaxed")
    await create_session_track(db_session, session.id, loud.id, track_order=3, dominant_state="excited")

    result = await session_tools.find_relaxing_tracks(make_run_context(make_deps(db_session)))

    assert sorted(t["title"] for t in result["tracks"]) == ["Gymnopedie", "Weightless"]
    assert result["count"] == 2
    assert result["total_available"] == 2


async def test_get_model_info_describes_the_held_model(db_session: AsyncSession) -> None:
    model = EEGNetClassifier()

    with override_eeg_model(model):
        info = json.loads(await session_tools.get_model_info(make_run_context(make_deps(db_session))))

    assert info["architecture"] == "EEGNet Dual-Head Classifier"
    assert info["parameters"]["total"] == sum(p.numel() for p in model.parameters())


async def test_get_model_info_without_a_model_fails(db_session: AsyncSession) -> None:
    with override_eeg_model(None), pytest.raises(ToolFailed, match="download-checkpoints.sh"):
        await session_tools.get_model_info(make_run_context(make_deps(db_session)))
