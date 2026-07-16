"""Test-data builders shared across test tiers.

``make_*`` returns an unpersisted ORM instance (unit tier). ``create_*``
persists via the given session — add, flush, refresh — for the integration
tier, where the surrounding fixture rolls an outer transaction back after
each test. Builders flush and never commit, same as services: the session's
owner holds the only commit.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.models.eeg_segment import EegSegment
from cortexdj.models.session import Session
from cortexdj.models.thread import Thread
from cortexdj.models.track_audio_embedding import EMBEDDING_DIM, TrackAudioEmbedding
from cortexdj.schemas.agent_type import AgentType


def make_session(**overrides: Any) -> Session:
    data: dict[str, Any] = {
        "id": str(uuid4()),
        "participant_id": "P01",
        "dataset_source": "deap",
        "recorded_at": datetime(2024, 1, 1, tzinfo=UTC),
        "duration_seconds": 60.0,
    }
    data.update(overrides)
    return Session(**data)


async def create_session(db: AsyncSession, **overrides: Any) -> Session:
    session = make_session(**overrides)
    db.add(session)
    await db.flush()
    await db.refresh(session)
    return session


def make_eeg_segment(session_id: str, **overrides: Any) -> EegSegment:
    data: dict[str, Any] = {
        "id": str(uuid4()),
        "session_id": session_id,
        "segment_index": 0,
        "start_time": 0.0,
        "end_time": 4.0,
        "arousal_score": 0.7,
        "valence_score": 0.6,
        "dominant_state": "excited",
        "band_powers": {"alpha": 1.0, "beta": 0.5, "theta": 0.3, "delta": 0.2, "gamma": 0.1},
    }
    data.update(overrides)
    return EegSegment(**data)


async def create_eeg_segment(db: AsyncSession, session_id: str, **overrides: Any) -> EegSegment:
    segment = make_eeg_segment(session_id, **overrides)
    db.add(segment)
    await db.flush()
    await db.refresh(segment)
    return segment


def make_track_audio_embedding(**overrides: Any) -> TrackAudioEmbedding:
    # Default is a unit vector, not zeros — cosine distance against a zero
    # vector is undefined, which would poison similarity-ordering tests.
    embedding = [0.0] * EMBEDDING_DIM
    embedding[0] = 1.0
    data: dict[str, Any] = {
        "spotify_id": f"sp-{uuid4().hex[:16]}",
        "title": "Test Track",
        "artist": "Test Artist",
        "source": "user_library",
        "embedding": embedding,
    }
    data.update(overrides)
    return TrackAudioEmbedding(**data)


async def create_track_audio_embedding(db: AsyncSession, **overrides: Any) -> TrackAudioEmbedding:
    row = make_track_audio_embedding(**overrides)
    db.add(row)
    await db.flush()
    await db.refresh(row)
    return row


def make_thread(**overrides: Any) -> Thread:
    data: dict[str, Any] = {
        "thread_id": f"t-{uuid4().hex[:8]}",
        "agent_type": AgentType.CHAT.value,
        "title": None,
    }
    data.update(overrides)
    return Thread(**data)


async def create_thread(db: AsyncSession, **overrides: Any) -> Thread:
    thread = make_thread(**overrides)
    db.add(thread)
    await db.flush()
    await db.refresh(thread)
    return thread
