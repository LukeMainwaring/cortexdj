from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from cortexdj.models.base import Base

_SINGLETON_ID = 1


class SpotifyToken(Base):
    """Stores Spotify OAuth tokens using a singleton pattern (always row id=1)
    since CortexDJ has no user authentication.
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
        result = await db.execute(select(cls).where(cls.id == _SINGLETON_ID))
        return result.scalar_one_or_none()

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
