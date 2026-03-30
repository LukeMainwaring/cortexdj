"""Spotify track model."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from sqlalchemy import DateTime, String, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from cortexdj.models.base import Base


class Track(Base):
    """Track metadata from Spotify or dataset stimulus info."""

    __tablename__ = "tracks"

    id: Mapped[str] = mapped_column(primary_key=True, index=True)
    spotify_track_id: Mapped[str | None] = mapped_column(String(255), unique=True)
    title: Mapped[str] = mapped_column(String(500))
    artist: Mapped[str] = mapped_column(String(500))
    album: Mapped[str | None] = mapped_column(String(500))
    duration_ms: Mapped[int | None]
    spotify_features: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    @classmethod
    async def get(cls, db: AsyncSession, track_id: str) -> Track | None:
        """Get a track by ID."""
        result = await db.execute(select(cls).where(cls.id == track_id))
        return result.scalar_one_or_none()

    @classmethod
    async def get_by_spotify_id(cls, db: AsyncSession, spotify_track_id: str) -> Track | None:
        """Get a track by Spotify ID."""
        result = await db.execute(select(cls).where(cls.spotify_track_id == spotify_track_id))
        return result.scalar_one_or_none()

    @classmethod
    async def get_many(cls, db: AsyncSession, track_ids: list[str]) -> Sequence[Track]:
        """Get multiple tracks by ID."""
        if not track_ids:
            return []
        result = await db.execute(select(cls).where(cls.id.in_(track_ids)))
        return result.scalars().all()

    @classmethod
    async def get_all(cls, db: AsyncSession, *, limit: int = 50) -> Sequence[Track]:
        """Get all tracks."""
        result = await db.execute(select(cls).order_by(cls.created_at.desc()).limit(limit))
        return result.scalars().all()
