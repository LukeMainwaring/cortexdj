"""Session management service."""

from __future__ import annotations

from typing import Any, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.models.eeg_segment import EegSegment
from cortexdj.models.session import Session
from cortexdj.models.session_track import SessionTrack
from cortexdj.models.track import Track


async def get_session(db: AsyncSession, session_id: str) -> Session | None:
    """Get a session by ID."""
    return await Session.get(db, session_id)


async def list_sessions(db: AsyncSession, *, limit: int = 50, offset: int = 0) -> tuple[Sequence[Session], int]:
    """List all sessions with pagination."""
    return await Session.get_all(db, limit=limit, offset=offset)


async def get_session_detail(db: AsyncSession, session_id: str) -> dict[str, Any] | None:
    """Get session with segment summary and track associations."""
    session = await Session.get(db, session_id)
    if session is None:
        return None

    summary = await EegSegment.get_session_summary(db, session_id)
    session_tracks = await SessionTrack.get_by_session(db, session_id)

    # Resolve track metadata
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
        "tracks": track_details,
    }
