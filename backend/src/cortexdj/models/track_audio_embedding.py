from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, String, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from cortexdj.models.base import Base

EMBEDDING_DIM = 512


class TrackAudioEmbedding(Base):
    __tablename__ = "track_audio_embeddings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    spotify_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    itunes_track_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    itunes_preview_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    title: Mapped[str] = mapped_column(String(512))
    artist: Mapped[str] = mapped_column(String(512))
    source: Mapped[str] = mapped_column(String(32))
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    @classmethod
    async def upsert(
        cls,
        db: AsyncSession,
        *,
        spotify_id: str,
        title: str,
        artist: str,
        source: str,
        embedding: np.ndarray,
        itunes_track_id: str | None = None,
        itunes_preview_url: str | None = None,
    ) -> None:
        vec = embedding.astype(np.float32).tolist()
        values = {
            "spotify_id": spotify_id,
            "title": title,
            "artist": artist,
            "source": source,
            "embedding": vec,
            "itunes_track_id": itunes_track_id,
            "itunes_preview_url": itunes_preview_url,
        }
        stmt = pg_insert(cls).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[cls.spotify_id],
            set_={
                "title": stmt.excluded.title,
                "artist": stmt.excluded.artist,
                "source": stmt.excluded.source,
                "embedding": stmt.excluded.embedding,
                "itunes_track_id": stmt.excluded.itunes_track_id,
                "itunes_preview_url": stmt.excluded.itunes_preview_url,
            },
        )
        await db.execute(stmt)

    @classmethod
    async def get_top_k_similar(
        cls, db: AsyncSession, query: np.ndarray, k: int = 10
    ) -> Sequence[tuple[TrackAudioEmbedding, float]]:
        query_vec = query.astype(np.float32).tolist()
        distance = cls.embedding.cosine_distance(query_vec).label("distance")
        stmt = select(cls, distance).order_by(distance).limit(k)
        result = await db.execute(stmt)
        return [(row, float(dist)) for row, dist in result.all()]

    @classmethod
    async def count(cls, db: AsyncSession) -> int:
        result = await db.execute(select(func.count(cls.id)))
        return int(result.scalar() or 0)
