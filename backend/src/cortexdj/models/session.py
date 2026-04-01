from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from sqlalchemy import DateTime, String, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from cortexdj.models.base import Base


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(primary_key=True, index=True)
    participant_id: Mapped[str] = mapped_column(String(50))
    dataset_source: Mapped[str] = mapped_column(String(20))
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[float]
    metadata_extra: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    @classmethod
    async def get(cls, db: AsyncSession, session_id: str) -> Session | None:
        result = await db.execute(select(cls).where(cls.id == session_id))
        return result.scalar_one_or_none()

    @classmethod
    async def get_all(cls, db: AsyncSession, *, limit: int = 50, offset: int = 0) -> tuple[Sequence[Session], int]:
        count_result = await db.execute(select(func.count(cls.id)))
        total = count_result.scalar() or 0
        result = await db.execute(select(cls).order_by(cls.recorded_at.desc()).limit(limit).offset(offset))
        return result.scalars().all(), total

    @classmethod
    async def get_by_participant(cls, db: AsyncSession, participant_id: str) -> Sequence[Session]:
        result = await db.execute(
            select(cls).where(cls.participant_id == participant_id).order_by(cls.recorded_at.desc())
        )
        return result.scalars().all()
