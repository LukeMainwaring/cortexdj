from typing import Any, Literal, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.models.eeg_segment import EegSegment
from cortexdj.models.session import Session
from cortexdj.models.session_track import SessionTrack
from cortexdj.models.track import Track
from cortexdj.services.trajectory import compute_trajectory_summary

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


async def list_sessions(db: AsyncSession, *, limit: int = 50, offset: int = 0) -> tuple[Sequence[Session], int]:
    return await Session.get_all(db, limit=limit, offset=offset)


async def list_sessions_enriched(
    db: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
    order: Literal["recent", "stable"] = "recent",
) -> tuple[list[dict[str, Any]], int]:
    """Return sessions enriched with display index, dominant-state label, and segment stats.

    display_index is a stable 1-based ordinal computed from chronological
    insertion order, so "Session 01" remains "Session 01" across requests as
    long as no rows are deleted. The display index is always assigned ASC by
    `created_at` regardless of the `order` argument, which only controls the
    order rows are returned in.

    order:
      - "recent" (default): newest first. `limit=1` returns the most recently
        recorded session — the right answer for prompts like "show me my most
        recent EEG session".
      - "stable": oldest first (Session 01 → Session NN). Use when the caller
        wants a deterministic catalog walk.
    """
    chronological = await Session.get_chronological_ids(db)
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
    weighted_arousal: dict[str, float] = {sid: 0.0 for sid in page_ids}
    weighted_valence: dict[str, float] = {sid: 0.0 for sid in page_ids}
    seg_totals: dict[str, int] = {sid: 0 for sid in page_ids}
    for sid, dominant_state, n, avg_arousal, avg_valence in await EegSegment.get_state_aggregates(db, page_ids):
        state_counts[sid][dominant_state] = n
        weighted_arousal[sid] += avg_arousal * n
        weighted_valence[sid] += avg_valence * n
        seg_totals[sid] += n

    track_counts = await SessionTrack.get_distinct_track_counts(db, page_ids)

    summaries: list[dict[str, Any]] = []
    for sid in page_ids:
        counts = state_counts.get(sid, {})
        total_segs = seg_totals.get(sid, 0) or 1
        distribution = {state: count / total_segs for state, count in counts.items()}
        dominant_state, dominant_fraction = (
            max(distribution.items(), key=lambda kv: kv[1]) if distribution else ("unknown", 0.0)
        )
        summaries.append(
            {
                "id": sid,
                "display_index": index_by_id[sid],
                "label": _build_label(dominant_state, dominant_fraction),
                "dominant_state": dominant_state,
                "state_distribution": distribution,
                "segment_count": seg_totals.get(sid, 0),
                "track_count": track_counts.get(sid, 0),
                "duration_seconds": duration_by_id[sid],
                "avg_arousal": (weighted_arousal[sid] / seg_totals[sid]) if seg_totals.get(sid) else 0.0,
                "avg_valence": (weighted_valence[sid] / seg_totals[sid]) if seg_totals.get(sid) else 0.0,
            }
        )

    return summaries, total


async def get_session_detail(db: AsyncSession, session_id: str) -> dict[str, Any] | None:
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
        # agent context but isn't cited by SessionCapability's narration
        # instructions. The frontend fetches it separately via
        # GET /sessions/{id}/segments for the trajectory chart.
        "trajectory_summary": (
            trajectory_summary.model_dump(mode="json", exclude={"smoothed"}) if trajectory_summary else None
        ),
        "tracks": track_details,
    }
