from fastapi import APIRouter, HTTPException

from cortexdj.dependencies.db import AsyncPostgresSessionDep
from cortexdj.models.eeg_segment import EegSegment
from cortexdj.schemas.eeg_segment import SegmentListResponse, SegmentSchema
from cortexdj.schemas.session import (
    SessionListResponse,
    SessionSchema,
    SessionSummaryListResponse,
    SessionSummarySchema,
)
from cortexdj.services import session as session_service
from cortexdj.services.trajectory import compute_trajectory_summary

sessions_router = APIRouter(prefix="/sessions", tags=["sessions"])


@sessions_router.get("")
async def list_sessions(db: AsyncPostgresSessionDep, limit: int = 50, offset: int = 0) -> SessionListResponse:
    """List all EEG sessions."""
    sessions, total = await session_service.list_sessions(db, limit=limit, offset=offset)
    return SessionListResponse(
        sessions=[SessionSchema.model_validate(s) for s in sessions],
        total=total,
    )


@sessions_router.get("/enriched")
async def list_sessions_enriched(
    db: AsyncPostgresSessionDep, limit: int = 50, offset: int = 0
) -> SessionSummaryListResponse:
    """List EEG sessions with derived display labels and quadrant distributions."""
    summaries, total = await session_service.list_sessions_enriched(db, limit=limit, offset=offset)
    return SessionSummaryListResponse(
        sessions=[SessionSummarySchema.model_validate(s) for s in summaries],
        total=total,
    )


@sessions_router.get("/{session_id}")
async def get_session(db: AsyncPostgresSessionDep, session_id: str) -> SessionSchema:
    """Get a specific EEG session."""
    session = await session_service.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionSchema.model_validate(session)


@sessions_router.get("/{session_id}/segments")
async def get_session_segments(db: AsyncPostgresSessionDep, session_id: str) -> SegmentListResponse:
    """Get all EEG segments for a session, plus a trajectory summary."""
    segments = await EegSegment.get_by_session(db, session_id)
    return SegmentListResponse(
        segments=[SegmentSchema.model_validate(s) for s in segments],
        total=len(segments),
        trajectory_summary=compute_trajectory_summary(segments),
    )
