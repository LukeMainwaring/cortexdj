"""Retrieval-index model classmethods against the real (migrated) schema.

The upsert exercises the unique index on ``spotify_id``.
"""

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.features.retrieval.model import EMBEDDING_DIM, TrackAudioEmbedding


class TestTrackAudioEmbeddingPgvector:
    async def test_upsert_replaces_on_spotify_id_conflict(self, db_session: AsyncSession) -> None:
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        vec[0] = 1.0
        await TrackAudioEmbedding.upsert(
            db_session, spotify_id="sp-up", title="Old", artist="A", source="user_library", embedding=vec
        )
        await TrackAudioEmbedding.upsert(
            db_session, spotify_id="sp-up", title="New", artist="A", source="user_library", embedding=vec
        )

        assert await TrackAudioEmbedding.count(db_session) == 1
        hits = await TrackAudioEmbedding.search_similar(db_session, vec, k=1)
        assert hits[0][0].title == "New"
