from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, distinct, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from cortexdj.shared.model_base import Base


class Session(Base):
    """One EEG recording: a participant listening to a run of stimulus tracks."""

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
    async def list_chronological_ids(cls, db: AsyncSession) -> list[tuple[str, float]]:
        """Return all (id, duration_seconds) pairs ordered by insertion time.

        Used to assign stable display indices ("Session 01"…"Session NN")
        independent of pagination.
        """
        result = await db.execute(select(cls.id, cls.duration_seconds).order_by(cls.created_at.asc()))
        return [(row.id, float(row.duration_seconds)) for row in result.all()]


class Track(Base):
    """A stimulus track; ``spotify_track_id`` is filled in lazily when a playlist first resolves it."""

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
        result = await db.execute(select(cls).where(cls.id == track_id))
        return result.scalar_one_or_none()

    @classmethod
    async def get_by_spotify_id(cls, db: AsyncSession, spotify_track_id: str) -> Track | None:
        result = await db.execute(select(cls).where(cls.spotify_track_id == spotify_track_id))
        return result.scalar_one_or_none()

    @classmethod
    async def get_many(cls, db: AsyncSession, track_ids: Sequence[str]) -> Sequence[Track]:
        if not track_ids:
            return []
        result = await db.execute(select(cls).where(cls.id.in_(track_ids)))
        return result.scalars().all()


class SessionTrack(Base):
    """A track as heard within one session, with the listener's response averaged over its segments."""

    __tablename__ = "session_tracks"

    id: Mapped[str] = mapped_column(primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"))
    track_id: Mapped[str] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"))
    track_order: Mapped[int]
    avg_arousal: Mapped[float]
    avg_valence: Mapped[float]
    dominant_state: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    @classmethod
    async def get_by_session(cls, db: AsyncSession, session_id: str) -> Sequence[SessionTrack]:
        result = await db.execute(select(cls).where(cls.session_id == session_id).order_by(cls.track_order.asc()))
        return result.scalars().all()

    @classmethod
    async def get_by_state(cls, db: AsyncSession, dominant_state: str, *, limit: int = 50) -> Sequence[SessionTrack]:
        """One row per distinct track heard in ``dominant_state``, at most ``limit`` tracks.

        Distinct before the limit: every session plays the same stimulus tracks, so a limit
        on raw rows would spend itself on repeats of a few tracks.
        """
        result = await db.execute(
            select(cls)
            .where(cls.dominant_state == dominant_state)
            .distinct(cls.track_id)
            .order_by(cls.track_id, cls.created_at)
            .limit(limit)
        )
        return result.scalars().all()

    @classmethod
    async def get_distinct_track_counts(cls, db: AsyncSession, session_ids: Sequence[str]) -> dict[str, int]:
        """Count distinct tracks per session for the given session ids."""
        if not session_ids:
            return {}
        result = await db.execute(
            select(cls.session_id, func.count(distinct(cls.track_id)))
            .where(cls.session_id.in_(session_ids))
            .group_by(cls.session_id)
        )
        return {sid: int(count) for sid, count in result.all()}


class EegSegment(Base):
    """A fixed-length EEG window within a session, classified into an arousal/valence quadrant."""

    __tablename__ = "eeg_segments"

    id: Mapped[str] = mapped_column(primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"))
    segment_index: Mapped[int]
    start_time: Mapped[float]
    end_time: Mapped[float]
    arousal_score: Mapped[float]
    valence_score: Mapped[float]
    dominant_state: Mapped[str] = mapped_column(String(20))
    band_powers: Mapped[dict[str, float]] = mapped_column(JSONB)
    features: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    @classmethod
    async def get_by_session(cls, db: AsyncSession, session_id: str) -> Sequence[EegSegment]:
        result = await db.execute(select(cls).where(cls.session_id == session_id).order_by(cls.segment_index.asc()))
        return result.scalars().all()

    @classmethod
    def summarize_segments(cls, session_id: str, segments: Sequence[EegSegment]) -> dict[str, Any] | None:
        """Aggregate a pre-fetched segment list into a session-level summary; None when there are no segments."""
        if not segments:
            return None

        arousal_scores = [s.arousal_score for s in segments]
        valence_scores = [s.valence_score for s in segments]
        state_counts: dict[str, int] = {}
        for s in segments:
            state_counts[s.dominant_state] = state_counts.get(s.dominant_state, 0) + 1

        dominant = max(state_counts, key=lambda state: state_counts[state])
        return {
            "session_id": session_id,
            "segment_count": len(segments),
            "avg_arousal": sum(arousal_scores) / len(arousal_scores),
            "avg_valence": sum(valence_scores) / len(valence_scores),
            "dominant_state": dominant,
            "state_distribution": state_counts,
            "duration_seconds": segments[-1].end_time - segments[0].start_time,
        }

    @classmethod
    async def get_session_summary(cls, db: AsyncSession, session_id: str) -> dict[str, Any] | None:
        segments = await cls.get_by_session(db, session_id)
        return cls.summarize_segments(session_id, segments)

    @classmethod
    async def get_state_aggregates(
        cls, db: AsyncSession, session_ids: Sequence[str]
    ) -> list[tuple[str, str, int, float, float]]:
        """Aggregate segments by (session_id, dominant_state).

        Returns rows of (session_id, dominant_state, n_segments, avg_arousal, avg_valence)
        for use by the enriched session list — avoids fetching every segment row.
        """
        if not session_ids:
            return []
        result = await db.execute(
            select(
                cls.session_id,
                cls.dominant_state,
                func.count().label("n_segments"),
                func.avg(cls.arousal_score).label("avg_arousal"),
                func.avg(cls.valence_score).label("avg_valence"),
            )
            .where(cls.session_id.in_(session_ids))
            .group_by(cls.session_id, cls.dominant_state)
        )
        return [
            (row.session_id, row.dominant_state, int(row.n_segments), float(row.avg_arousal), float(row.avg_valence))
            for row in result.all()
        ]


class ThreadBrainContext(Base):
    """The brain state a conversation is focused on, set by the ``set_brain_context`` tool.

    Its own table keyed by thread, not a column on ``threads``: the chat slice owns that
    table and knows nothing of EEG. The foreign key names ``threads`` by string so no Python
    import crosses into ``features/chat``; deleting a thread deletes its context. The thread
    row always exists before a tool runs (the chat stream creates it), so do not create
    threads from here.
    """

    __tablename__ = "thread_brain_contexts"

    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.thread_id", ondelete="CASCADE"), primary_key=True)
    latest_session_id: Mapped[str | None]
    dominant_mood: Mapped[str | None]
    avg_arousal: Mapped[float | None]
    avg_valence: Mapped[float | None]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    @classmethod
    async def get(cls, db: AsyncSession, thread_id: str) -> ThreadBrainContext | None:
        return await db.get(cls, thread_id)

    @classmethod
    async def update_fields(cls, db: AsyncSession, thread_id: str, fields: dict[str, Any]) -> ThreadBrainContext:
        """Overwrite only the given columns, creating the row on the thread's first context.

        The write runs in a savepoint: the session is the agent run's shared one, and a
        failed insert (no such thread) would otherwise leave it unusable for every later tool.
        """
        async with db.begin_nested():
            context = await cls.get(db, thread_id)
            if context is None:
                context = cls(thread_id=thread_id)
                db.add(context)
            for name, value in fields.items():
                setattr(context, name, value)
            await db.flush()
        await db.refresh(context)
        return context
