"""Retrieval-index model classmethods against the real (migrated) schema.

The cosine ordering exercises the pgvector extension and the HNSW index the migration
creates; the upsert exercises the unique index on ``spotify_id``.
"""

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.features.retrieval.model import EMBEDDING_DIM, TrackAudioEmbedding
from tests.factories import create_track_audio_embedding


class TestTrackAudioEmbeddingPgvector:
    async def test_cosine_search_orders_nearest_first(self, db_session: AsyncSession) -> None:
        e1 = [0.0] * EMBEDDING_DIM
        e1[0] = 1.0
        e2 = [0.0] * EMBEDDING_DIM
        e2[1] = 1.0
        await create_track_audio_embedding(db_session, spotify_id="sp-near", embedding=e1)
        await create_track_audio_embedding(db_session, spotify_id="sp-far", embedding=e2)

        hits = await TrackAudioEmbedding.search_similar(db_session, np.array(e1, dtype=np.float32), k=2)

        assert [row.spotify_id for row, _ in hits] == ["sp-near", "sp-far"]
        near_distance, far_distance = hits[0][1], hits[1][1]
        assert near_distance == pytest.approx(0.0, abs=1e-5)
        assert far_distance == pytest.approx(1.0, abs=1e-5)

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
