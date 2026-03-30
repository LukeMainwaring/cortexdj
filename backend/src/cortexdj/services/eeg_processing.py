"""EEG processing service — orchestrates preprocessing, classification, and session analysis."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.models.eeg_segment import EegSegment
from cortexdj.models.session import Session
from cortexdj.models.session_track import SessionTrack
from cortexdj.models.track import Track

logger = logging.getLogger(__name__)


async def get_session_analysis(db: AsyncSession, session_id: str) -> dict[str, Any]:
    """Get full analysis of a session including segment breakdown and tracks."""
    session = await Session.get(db, session_id)
    if session is None:
        return {"error": f"Session {session_id} not found."}

    summary = await EegSegment.get_session_summary(db, session_id)
    segments = await EegSegment.get_by_session(db, session_id)

    segment_details = [
        {
            "index": s.segment_index,
            "time_range": f"{s.start_time:.1f}s - {s.end_time:.1f}s",
            "arousal": s.arousal_score,
            "valence": s.valence_score,
            "state": s.dominant_state,
            "band_powers": s.band_powers,
        }
        for s in segments
    ]

    # Get associated tracks
    session_tracks = await SessionTrack.get_by_session(db, session_id)
    track_info = []
    for st in session_tracks:
        track = await Track.get(db, st.track_id)
        if track:
            track_info.append({
                "title": track.title,
                "artist": track.artist,
                "dominant_state": st.dominant_state,
                "avg_arousal": st.avg_arousal,
                "avg_valence": st.avg_valence,
            })

    return {
        "session_id": session_id,
        "participant": session.participant_id,
        "source": session.dataset_source,
        "summary": summary,
        "segments": segment_details,
        "tracks": track_info,
    }


async def compare_sessions(
    db: AsyncSession, session_id_1: str, session_id_2: str
) -> dict[str, Any]:
    """Compare brain state patterns across two sessions."""
    summary_1 = await EegSegment.get_session_summary(db, session_id_1)
    summary_2 = await EegSegment.get_session_summary(db, session_id_2)

    if "error" in summary_1 or "error" in summary_2:
        return {"error": "One or both sessions not found or have no segments."}

    return {
        "session_1": summary_1,
        "session_2": summary_2,
        "comparison": {
            "arousal_delta": summary_1["avg_arousal"] - summary_2["avg_arousal"],
            "valence_delta": summary_1["avg_valence"] - summary_2["avg_valence"],
            "same_dominant_state": summary_1["dominant_state"] == summary_2["dominant_state"],
        },
    }


async def find_tracks_by_mood(
    db: AsyncSession, target_state: str, *, limit: int = 20
) -> list[dict[str, Any]]:
    """Find tracks that triggered a specific brain state."""
    session_tracks = await SessionTrack.get_by_state(db, target_state, limit=limit)

    results = []
    seen_track_ids: set[str] = set()
    for st in session_tracks:
        if st.track_id in seen_track_ids:
            continue
        seen_track_ids.add(st.track_id)

        track = await Track.get(db, st.track_id)
        if track:
            results.append({
                "track_id": track.id,
                "title": track.title,
                "artist": track.artist,
                "avg_arousal": st.avg_arousal,
                "avg_valence": st.avg_valence,
                "dominant_state": st.dominant_state,
            })

    return results
