"""CRUD classmethod tests against the real (migrated) schema.

Covers the model behaviors the HTTP tier doesn't reach directly: pagination
semantics, idempotent get_or_create, append-only message history, the
Spotify token singleton, and a pgvector cosine-ordering smoke test that
exercises the extension + HNSW index created by the migrations.
"""

from datetime import UTC, datetime

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.models.message import Message
from cortexdj.models.session import Session
from cortexdj.models.spotify_token import SpotifyToken
from cortexdj.models.thread import Thread
from cortexdj.models.track_audio_embedding import EMBEDDING_DIM, TrackAudioEmbedding
from cortexdj.schemas.agent_type import AgentType
from cortexdj.schemas.thread import BrainContext
from tests.factories import create_session, create_track_audio_embedding

_CHAT = AgentType.CHAT.value


class TestSession:
    async def test_get_all_paginates_newest_first(self, db_session: AsyncSession) -> None:
        for day in (1, 2, 3):
            await create_session(db_session, recorded_at=datetime(2024, 2, day, tzinfo=UTC))

        sessions, total = await Session.get_all(db_session, limit=2, offset=0)
        assert total == 3
        assert len(sessions) == 2
        assert sessions[0].recorded_at.day == 3

    async def test_get_by_participant(self, db_session: AsyncSession) -> None:
        await create_session(db_session, participant_id="P01")
        await create_session(db_session, participant_id="P02")

        found = await Session.get_by_participant(db_session, "P02")
        assert [s.participant_id for s in found] == ["P02"]


class TestThread:
    async def test_get_or_create_is_idempotent(self, db_session: AsyncSession) -> None:
        first = await Thread.get_or_create(db_session, "t-idem", _CHAT)
        second = await Thread.get_or_create(db_session, "t-idem", _CHAT)
        assert first.thread_id == second.thread_id
        assert len(await Thread.list_all(db_session, _CHAT)) == 1

    async def test_update_title(self, db_session: AsyncSession) -> None:
        await Thread.get_or_create(db_session, "t-title", _CHAT)
        await Thread.update_title(db_session, "t-title", _CHAT, "Morning focus")

        thread = await Thread.get(db_session, "t-title", _CHAT)
        assert thread is not None
        assert thread.title == "Morning focus"

    async def test_update_brain_context_merges(self, db_session: AsyncSession) -> None:
        # Creates the thread on first write, then merges only set fields.
        merged = await Thread.update_brain_context(
            db_session, "t-ctx", _CHAT, BrainContext(latest_session_id="sess-1", dominant_mood="relaxed")
        )
        assert merged.latest_session_id == "sess-1"

        merged = await Thread.update_brain_context(db_session, "t-ctx", _CHAT, BrainContext(dominant_mood="excited"))
        assert merged.dominant_mood == "excited"
        assert merged.latest_session_id == "sess-1", "unset fields must survive the merge"

    async def test_delete_by_id(self, db_session: AsyncSession) -> None:
        await Thread.get_or_create(db_session, "t-del", _CHAT)
        await Thread.delete_by_id(db_session, "t-del", _CHAT)
        assert await Thread.get(db_session, "t-del", _CHAT) is None


class TestMessage:
    async def test_save_history_is_append_only(self, db_session: AsyncSession) -> None:
        await Thread.get_or_create(db_session, "t-msg", _CHAT)
        first_batch: list[dict[str, object]] = [{"kind": "request", "n": 1}, {"kind": "response", "n": 2}]
        await Message.save_history(db_session, "t-msg", _CHAT, first_batch)

        # Re-saving the full history plus one new message must insert only
        # the new one — existing rows keep their ids and timestamps.
        await Message.save_history(db_session, "t-msg", _CHAT, [*first_batch, {"kind": "request", "n": 3}])

        history = await Message.get_history(db_session, "t-msg", _CHAT)
        assert [m["n"] for m in history] == [1, 2, 3]


class TestSpotifyToken:
    async def test_upsert_then_update_keeps_singleton(self, db_session: AsyncSession) -> None:
        expires = datetime(2030, 1, 1, tzinfo=UTC)
        assert not await SpotifyToken.is_connected(db_session)

        await SpotifyToken.upsert(db_session, "access-1", "refresh-1", expires)
        token = await SpotifyToken.upsert(db_session, "access-2", "refresh-2", expires)

        assert token.access_token == "access-2"
        assert await SpotifyToken.is_connected(db_session)

    async def test_clear(self, db_session: AsyncSession) -> None:
        await SpotifyToken.upsert(db_session, "a", "r", datetime(2030, 1, 1, tzinfo=UTC))
        await SpotifyToken.clear(db_session)
        assert not await SpotifyToken.is_connected(db_session)


class TestTrackAudioEmbeddingPgvector:
    async def test_cosine_search_orders_nearest_first(self, db_session: AsyncSession) -> None:
        e1 = [0.0] * EMBEDDING_DIM
        e1[0] = 1.0
        e2 = [0.0] * EMBEDDING_DIM
        e2[1] = 1.0
        await create_track_audio_embedding(db_session, spotify_id="sp-near", embedding=e1)
        await create_track_audio_embedding(db_session, spotify_id="sp-far", embedding=e2)

        hits = await TrackAudioEmbedding.get_top_k_similar(db_session, np.array(e1, dtype=np.float32), k=2)

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
        hits = await TrackAudioEmbedding.get_top_k_similar(db_session, vec, k=1)
        assert hits[0][0].title == "New"
