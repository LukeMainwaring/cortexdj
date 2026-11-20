from collections.abc import Sequence
from typing import Any, Literal

from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.features.sessions.model import EegSegment, Session, SessionTrack, ThreadBrainContext, Track
from cortexdj.features.sessions.schema import BrainContext, SessionSummarySchema, TrajectorySummary
from cortexdj.features.sessions.trajectory import compute_trajectory_summary
from cortexdj.shared.track_access import MoodTrack, TrackInfo

_STATE_ADJECTIVES = {
    "relaxed": "Relaxed",
    "calm": "Calm & focused",
    "excited": "Excited",
    "stressed": "Tense",
}


def _build_label(dominant_state: str, dwell_fraction: float) -> str:
    base = _STATE_ADJECTIVES.get(dominant_state, dominant_state.title())
    if dwell_fraction >= 0.9:
        return f"{base} throughout"
    if dwell_fraction >= 0.6:
        return f"Mostly {base.lower()}"
    return f"Mixed, leaning {base.lower()}"


async def get_session(db: AsyncSession, session_id: str) -> Session | None:
    return await Session.get(db, session_id)


async def get_session_participant_id(db: AsyncSession, session_id: str) -> str | None:
    """The session's participant id; what ``api.py`` registers as the shared session lookup."""
    session = await Session.get(db, session_id)
    return session.participant_id if session is not None else None


async def list_sessions(db: AsyncSession, *, limit: int = 50, offset: int = 0) -> tuple[Sequence[Session], int]:
    return await Session.list_all(db, limit=limit, offset=offset)


async def list_sessions_enriched(
    db: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
    order: Literal["recent", "stable"] = "recent",
) -> tuple[list[SessionSummarySchema], int]:
    """Return sessions enriched with display index, dominant-state label, and segment stats.

    display_index is a 1-based ordinal computed from chronological insertion
    order; `order` only controls the returned ordering. Indices are *not*
    persisted, so a deletion renumbers every subsequent session — acceptable
    for the single-tenant demo but should become a real column before
    multi-user sessions land.
    """
    chronological = await Session.list_chronological_ids(db)
    total = len(chronological)
    if total == 0:
        return [], 0

    index_by_id: dict[str, int] = {sid: i + 1 for i, (sid, _) in enumerate(chronological)}
    duration_by_id: dict[str, float] = dict(chronological)

    ordered = list(reversed(chronological)) if order == "recent" else chronological
    page_ids = [sid for sid, _ in ordered[offset : offset + limit]]
    if not page_ids:
        return [], total

    state_counts: dict[str, dict[str, int]] = {sid: {} for sid in page_ids}
    weighted_arousal: dict[str, float] = dict.fromkeys(page_ids, 0.0)
    weighted_valence: dict[str, float] = dict.fromkeys(page_ids, 0.0)
    seg_totals: dict[str, int] = dict.fromkeys(page_ids, 0)
    for sid, dominant_state, n, avg_arousal, avg_valence in await EegSegment.get_state_aggregates(db, page_ids):
        state_counts[sid][dominant_state] = n
        weighted_arousal[sid] += avg_arousal * n
        weighted_valence[sid] += avg_valence * n
        seg_totals[sid] += n

    track_counts = await SessionTrack.get_distinct_track_counts(db, page_ids)

    summaries: list[SessionSummarySchema] = []
    for sid in page_ids:
        counts = state_counts.get(sid, {})
        total_segs = seg_totals.get(sid, 0) or 1
        distribution = {state: count / total_segs for state, count in counts.items()}
        dominant_state, dominant_fraction = (
            max(distribution.items(), key=lambda kv: kv[1]) if distribution else ("unknown", 0.0)
        )
        summaries.append(
            SessionSummarySchema(
                id=sid,
                display_index=index_by_id[sid],
                label=_build_label(dominant_state, dominant_fraction),
                dominant_state=dominant_state,
                state_distribution=distribution,
                segment_count=seg_totals.get(sid, 0),
                track_count=track_counts.get(sid, 0),
                duration_seconds=duration_by_id[sid],
                avg_arousal=(weighted_arousal[sid] / seg_totals[sid]) if seg_totals.get(sid) else 0.0,
                avg_valence=(weighted_valence[sid] / seg_totals[sid]) if seg_totals.get(sid) else 0.0,
            )
        )

    return summaries, total


async def get_session_detail(db: AsyncSession, session_id: str) -> dict[str, Any] | None:
    """The session, its segment summary, trajectory, and tracks as the ``analyze_session`` tool returns them.

    None when the session does not exist; ``summary`` and ``trajectory_summary`` are None
    for a session with no segments.
    """
    session = await Session.get(db, session_id)
    if session is None:
        return None

    segments = await EegSegment.get_by_session(db, session_id)
    summary = EegSegment.summarize_segments(session_id, segments)
    trajectory_summary = compute_trajectory_summary(segments)
    session_tracks = await SessionTrack.get_by_session(db, session_id)

    track_details = []
    for st in session_tracks:
        track = await Track.get(db, st.track_id)
        if track:
            track_details.append(
                {
                    "track_id": track.id,
                    "title": track.title,
                    "artist": track.artist,
                    "track_order": st.track_order,
                    "avg_arousal": st.avg_arousal,
                    "avg_valence": st.avg_valence,
                    "dominant_state": st.dominant_state,
                }
            )

    return {
        "session": {
            "id": session.id,
            "participant_id": session.participant_id,
            "dataset_source": session.dataset_source,
            "recorded_at": session.recorded_at.isoformat(),
            "duration_seconds": session.duration_seconds,
        },
        "summary": summary,
        # Exclude `smoothed` here: it's dense per-segment data that bloats the
        # agent context but isn't cited by SessionsCapability's narration
        # instructions. The frontend fetches it separately via
        # GET /sessions/{id}/segments for the trajectory chart.
        "trajectory_summary": (
            trajectory_summary.model_dump(mode="json", exclude={"smoothed"}) if trajectory_summary else None
        ),
        "tracks": track_details,
    }


async def get_session_segments(
    db: AsyncSession, session_id: str
) -> tuple[Sequence[EegSegment], TrajectorySummary | None]:
    """A session's segments in order with their trajectory summary; empty for an unknown session."""
    segments = await EegSegment.get_by_session(db, session_id)
    return segments, compute_trajectory_summary(segments)


async def get_session_summary(db: AsyncSession, session_id: str) -> dict[str, Any] | None:
    """A session's segment-level summary; None when it has no segments."""
    return await EegSegment.get_session_summary(db, session_id)


async def compare_sessions(db: AsyncSession, session_id_1: str, session_id_2: str) -> dict[str, Any] | None:
    """Both sessions' summaries and their deltas; None when either session has no segments."""
    summary_1 = await EegSegment.get_session_summary(db, session_id_1)
    summary_2 = await EegSegment.get_session_summary(db, session_id_2)

    if summary_1 is None or summary_2 is None:
        return None

    return {
        "session_1": summary_1,
        "session_2": summary_2,
        "comparison": {
            "arousal_delta": summary_1["avg_arousal"] - summary_2["avg_arousal"],
            "valence_delta": summary_1["avg_valence"] - summary_2["avg_valence"],
            "same_dominant_state": summary_1["dominant_state"] == summary_2["dominant_state"],
        },
    }


async def find_tracks_by_mood(db: AsyncSession, target_state: str, limit: int = 20) -> list[MoodTrack]:
    """Distinct tracks some session's listener responded to with ``target_state``."""
    session_tracks = await SessionTrack.get_by_state(db, target_state, limit=limit)

    results: list[MoodTrack] = []
    for st in session_tracks:
        track = await Track.get(db, st.track_id)
        if track:
            results.append(
                MoodTrack(
                    track_id=track.id,
                    title=track.title,
                    artist=track.artist,
                    dominant_state=st.dominant_state,
                    avg_arousal=st.avg_arousal,
                    avg_valence=st.avg_valence,
                )
            )

    return results


async def get_tracks(db: AsyncSession, track_ids: Sequence[str]) -> list[TrackInfo]:
    tracks = await Track.get_many(db, track_ids)
    return [TrackInfo(id=t.id, title=t.title, artist=t.artist, spotify_track_id=t.spotify_track_id) for t in tracks]


async def save_track_spotify_id(db: AsyncSession, track_id: str, spotify_track_id: str) -> None:
    """Store a resolved Spotify id on the track.

    ``spotify_track_id`` is unique, so the write is skipped when another track already holds
    the id; the caller can still use the id for this track.
    """
    track = await Track.get(db, track_id)
    if track is None or track.spotify_track_id == spotify_track_id:
        return
    if await Track.get_by_spotify_id(db, spotify_track_id) is None:
        track.spotify_track_id = spotify_track_id
        await db.flush()


async def get_brain_context(db: AsyncSession, thread_id: str) -> BrainContext | None:
    context = await ThreadBrainContext.get(db, thread_id)
    return BrainContext.model_validate(context) if context is not None else None


async def merge_brain_context(db: AsyncSession, thread_id: str, updates: BrainContext) -> BrainContext:
    """Merge ``updates`` into the thread's brain context: only fields explicitly set on it overwrite."""
    context = await ThreadBrainContext.update_fields(db, thread_id, updates.model_dump(exclude_unset=True))
    return BrainContext.model_validate(context)
