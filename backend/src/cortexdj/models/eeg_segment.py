from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from sqlalchemy import DateTime, ForeignKey, String, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from cortexdj.models.base import Base


class EegSegment(Base):
    __tablename__ = "eeg_segments"

    id: Mapped[str] = mapped_column(primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"))
    segment_index: Mapped[int]
    start_time: Mapped[float]
    end_time: Mapped[float]
    arousal_score: Mapped[float]
    valence_score: Mapped[float]
    dominant_state: Mapped[str] = mapped_column(String(20))
    band_powers: Mapped[dict[str, float]] = mapped_column(JSONB)
    features: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    @classmethod
    async def get_by_session(cls, db: AsyncSession, session_id: str) -> Sequence[EegSegment]:
        result = await db.execute(select(cls).where(cls.session_id == session_id).order_by(cls.segment_index.asc()))
        return result.scalars().all()

    @classmethod
    async def get_by_state(cls, db: AsyncSession, dominant_state: str, *, limit: int = 50) -> Sequence[EegSegment]:
        result = await db.execute(
            select(cls).where(cls.dominant_state == dominant_state).order_by(cls.created_at.desc()).limit(limit)
        )
        return result.scalars().all()

    @classmethod
    def summarize_segments(cls, session_id: str, segments: Sequence[EegSegment]) -> dict[str, Any]:
        """Aggregate a pre-fetched segment list into a session-level summary."""
        if not segments:
            return {"error": "No segments found"}

        arousal_scores = [s.arousal_score for s in segments]
        valence_scores = [s.valence_score for s in segments]
        state_counts: dict[str, int] = {}
        for s in segments:
            state_counts[s.dominant_state] = state_counts.get(s.dominant_state, 0) + 1

        dominant = max(state_counts, key=state_counts.get)  # type: ignore[arg-type]
        return {
            "session_id": session_id,
            "segment_count": len(segments),
            "avg_arousal": sum(arousal_scores) / len(arousal_scores),
            "avg_valence": sum(valence_scores) / len(valence_scores),
            "dominant_state": dominant,
            "state_distribution": state_counts,
            "duration_seconds": segments[-1].end_time - segments[0].start_time,
        }

    @classmethod
    async def get_session_summary(cls, db: AsyncSession, session_id: str) -> dict[str, Any]:
        segments = await cls.get_by_session(db, session_id)
        return cls.summarize_segments(session_id, segments)
