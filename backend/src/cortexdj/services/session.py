from __future__ import annotations

from typing import Any, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.models.eeg_segment import EegSegment
from cortexdj.models.session import Session
from cortexdj.models.session_track import SessionTrack
from cortexdj.models.track import Track
from cortexdj.services.trajectory import compute_trajectory_summary


async def get_session(db: AsyncSession, session_id: str) -> Session | None:
    return await Session.get(db, session_id)


async def list_sessions(db: AsyncSession, *, limit: int = 50, offset: int = 0) -> tuple[Sequence[Session], int]:
    return await Session.get_all(db, limit=limit, offset=offset)


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
