from fastapi import APIRouter

from cortexdj.core.db import AsyncPostgresSessionDep
from cortexdj.features.sessions import service as session_service
from cortexdj.features.sessions.schema import (
    BrainContext,
    SegmentListResponse,
    SegmentSchema,
    SessionSummaryListResponse,
)

sessions_router = APIRouter(prefix="/sessions", tags=["sessions"])
brain_context_router = APIRouter(prefix="/brain-context", tags=["sessions"])


@sessions_router.get("/enriched")
async def list_sessions_enriched(db: AsyncPostgresSessionDep, limit: int = 50) -> SessionSummaryListResponse:
    """List EEG sessions, newest first, with derived display labels and quadrant distributions."""
    summaries, total = await session_service.list_sessions_enriched(db, limit=limit)
    return SessionSummaryListResponse(sessions=summaries, total=total)


@sessions_router.get("/{session_id}/segments")
async def get_session_segments(db: AsyncPostgresSessionDep, session_id: str) -> SegmentListResponse:
    """Get all EEG segments for a session, plus a trajectory summary."""
    segments, trajectory_summary = await session_service.get_session_segments(db, session_id)
    return SegmentListResponse(
        segments=[SegmentSchema.model_validate(s) for s in segments],
        total=len(segments),
        trajectory_summary=trajectory_summary,
    )


@brain_context_router.get("/{thread_id}")
async def get_brain_context(db: AsyncPostgresSessionDep, thread_id: str) -> BrainContext | None:
    """Get the brain context a chat thread is focused on; null when none has been set."""
    return await session_service.get_brain_context(db, thread_id)
