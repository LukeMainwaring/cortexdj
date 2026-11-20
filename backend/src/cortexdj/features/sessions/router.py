from typing import Literal

from fastapi import APIRouter, HTTPException

from cortexdj.core.db import AsyncPostgresSessionDep
from cortexdj.features.sessions import service as session_service
from cortexdj.features.sessions.schema import (
    BrainContext,
    SegmentListResponse,
    SegmentSchema,
    SessionListResponse,
    SessionSchema,
    SessionSummaryListResponse,
)

sessions_router = APIRouter(prefix="/sessions", tags=["sessions"])
brain_context_router = APIRouter(prefix="/brain-context", tags=["sessions"])


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
    db: AsyncPostgresSessionDep,
    limit: int = 50,
    offset: int = 0,
    order: Literal["recent", "stable"] = "recent",
) -> SessionSummaryListResponse:
    """List EEG sessions with derived display labels and quadrant distributions.

    `order=recent` (default) returns newest first; `order=stable` returns
    chronological-insertion order (Session 01 → Session NN).
    """
    summaries, total = await session_service.list_sessions_enriched(db, limit=limit, offset=offset, order=order)
    return SessionSummaryListResponse(sessions=summaries, total=total)


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
