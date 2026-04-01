from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy import DateTime, ForeignKey, String, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from cortexdj.models.base import Base


class SessionTrack(Base):
    __tablename__ = "session_tracks"

    id: Mapped[str] = mapped_column(primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"))
    track_id: Mapped[str] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"))
    track_order: Mapped[int]
    avg_arousal: Mapped[float]
    avg_valence: Mapped[float]
    dominant_state: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    @classmethod
    async def get_by_session(cls, db: AsyncSession, session_id: str) -> Sequence[SessionTrack]:
        result = await db.execute(select(cls).where(cls.session_id == session_id).order_by(cls.track_order.asc()))
        return result.scalars().all()

    @classmethod
    async def get_by_state(cls, db: AsyncSession, dominant_state: str, *, limit: int = 50) -> Sequence[SessionTrack]:
        result = await db.execute(select(cls).where(cls.dominant_state == dominant_state).limit(limit))
        return result.scalars().all()

    @classmethod
    async def get_relaxing_tracks(cls, db: AsyncSession, *, limit: int = 50) -> Sequence[SessionTrack]:
        result = await db.execute(
            select(cls)
            .where(cls.avg_arousal < 0.5, cls.avg_valence >= 0.5)
            .order_by(cls.avg_valence.desc())
            .limit(limit)
        )
        return result.scalars().all()
