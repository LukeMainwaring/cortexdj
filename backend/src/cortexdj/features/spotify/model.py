from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, String, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from cortexdj.shared.model_base import Base

_SINGLETON_ID = 1


class SpotifyToken(Base):
    """The user's Spotify OAuth tokens, always the single row id=1: CortexDJ has no user accounts.

    ``updated_at`` is nullable with no server default, unlike the template's other
    timestamps; it matches the table as it exists, so do not tidy it without a migration.
    """

    __tablename__ = "spotify_tokens"

    id: Mapped[int] = mapped_column(primary_key=True, default=_SINGLETON_ID)
    access_token: Mapped[str] = mapped_column(String(1024))
    refresh_token: Mapped[str] = mapped_column(String(1024))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())

    @classmethod
    async def get(cls, db: AsyncSession) -> SpotifyToken | None:
        # A primary-key get, like ThreadBrainContext.get: the assistant reads the token on
        # every model step, and the unit tier's mock session answers db.get with no row.
        return await db.get(cls, _SINGLETON_ID)

    @classmethod
    async def upsert(
        cls,
        db: AsyncSession,
        access_token: str,
        refresh_token: str,
        expires_at: datetime,
    ) -> SpotifyToken:
        token = await cls.get(db)
        if token is None:
            token = cls(
                id=_SINGLETON_ID,
                access_token=access_token,
                refresh_token=refresh_token,
                expires_at=expires_at,
            )
            db.add(token)
        else:
            token.access_token = access_token
            token.refresh_token = refresh_token
            token.expires_at = expires_at
        await db.flush()
        return token

    @classmethod
    async def clear(cls, db: AsyncSession) -> None:
        token = await cls.get(db)
        if token is not None:
            await db.delete(token)
            await db.flush()

    @classmethod
    async def is_connected(cls, db: AsyncSession) -> bool:
        return await cls.get(db) is not None


class Playlist(Base):
    """A mood playlist the assistant built; ``spotify_playlist_id`` is None when it was saved only locally."""

    __tablename__ = "playlists"

    id: Mapped[str] = mapped_column(primary_key=True, index=True)
    spotify_playlist_id: Mapped[str | None] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255))
    mood_criteria: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    track_count: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    @classmethod
    async def get(cls, db: AsyncSession, playlist_id: str) -> Playlist | None:
        result = await db.execute(select(cls).where(cls.id == playlist_id))
        return result.scalar_one_or_none()

    @classmethod
    async def list_all(cls, db: AsyncSession, *, limit: int = 50) -> Sequence[Playlist]:
        result = await db.execute(select(cls).order_by(cls.created_at.desc()).limit(limit))
        return result.scalars().all()
