"""Agent-generated playlist model."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from sqlalchemy import DateTime, String, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from cortexdj.models.base import Base


class Playlist(Base):
    """Agent-generated Spotify playlist with brain-derived mood criteria."""

    __tablename__ = "playlists"

    id: Mapped[str] = mapped_column(primary_key=True, index=True)
    spotify_playlist_id: Mapped[str | None] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255))
    mood_criteria: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    track_count: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    @classmethod
    async def get(cls, db: AsyncSession, playlist_id: str) -> Playlist | None:
        """Get a playlist by ID."""
        result = await db.execute(select(cls).where(cls.id == playlist_id))
        return result.scalar_one_or_none()

    @classmethod
    async def get_all(cls, db: AsyncSession, *, limit: int = 50) -> Sequence[Playlist]:
        """Get all playlists."""
        result = await db.execute(select(cls).order_by(cls.created_at.desc()).limit(limit))
        return result.scalars().all()
