from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, delete, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from cortexdj.shared.model_base import Base

# Postgres SQLSTATE for a duplicate key.
_UNIQUE_VIOLATION = "23505"


class Thread(Base):
    """One conversation with exactly one message history.

    Keyed by ``thread_id`` alone on purpose: whichever named agent answers a
    turn, it reads and extends the same history, so agents share context across
    a conversation. Do not key history by agent.
    """

    __tablename__ = "threads"

    thread_id: Mapped[str] = mapped_column(primary_key=True)
    title: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    @classmethod
    async def get(cls, db: AsyncSession, thread_id: str) -> Thread | None:
        return await db.get(cls, thread_id)

    @classmethod
    async def create(cls, db: AsyncSession, thread_id: str) -> Thread:
        thread = cls(thread_id=thread_id)
        db.add(thread)
        await db.flush()
        await db.refresh(thread)
        return thread

    @classmethod
    async def get_or_create(cls, db: AsyncSession, thread_id: str) -> Thread:
        """The thread, started if this is its first turn; safe when two first turns race.

        The client picks the id before its first message, so two tabs (or a double
        submit) can both find no thread and both insert. The loser's primary-key
        violation is the race, not a bug: a savepoint keeps the request's session
        usable and the winner's row is returned. Any other constraint failure propagates.
        """
        thread = await cls.get(db, thread_id)
        if thread is not None:
            return thread
        try:
            async with db.begin_nested():
                return await cls.create(db, thread_id)
        except IntegrityError as exc:
            if getattr(exc.orig, "sqlstate", None) != _UNIQUE_VIOLATION:
                raise
        winner = await cls.get(db, thread_id)
        assert winner is not None  # the request that won the race inserted it
        return winner

    @classmethod
    async def list_all(cls, db: AsyncSession) -> list[Thread]:
        result = await db.execute(select(cls).order_by(cls.updated_at.desc()))
        return list(result.scalars().all())

    @classmethod
    async def delete_by_id(cls, db: AsyncSession, thread_id: str) -> bool:
        """Delete the thread (its messages cascade); False when there was no such thread."""
        result = await db.execute(delete(cls).where(cls.thread_id == thread_id).returning(cls.thread_id))
        await db.flush()
        return result.scalar_one_or_none() is not None

    @classmethod
    async def update_title(cls, db: AsyncSession, thread_id: str, title: str) -> Thread | None:
        thread = await cls.get(db, thread_id)
        if thread:
            thread.title = title
            await db.flush()
        return thread

    @classmethod
    async def touch(cls, db: AsyncSession, thread_id: str) -> Thread | None:
        """Mark the thread as active now, which moves it to the top of the thread list."""
        thread = await cls.get(db, thread_id)
        if thread:
            thread.updated_at = datetime.now(UTC)
            await db.flush()
        return thread


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.thread_id", ondelete="CASCADE"))
    message_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    @classmethod
    async def get_history(cls, db: AsyncSession, thread_id: str) -> list[dict[str, Any]]:
        result = await db.execute(select(cls.message_data).where(cls.thread_id == thread_id).order_by(cls.id.asc()))
        return [row[0] for row in result.all()]

    @classmethod
    async def append_messages(cls, db: AsyncSession, thread_id: str, messages: list[dict[str, Any]]) -> None:
        """Append messages to the thread history; the caller decides what is new."""
        for msg_data in messages:
            db.add(cls(thread_id=thread_id, message_data=msg_data))
        if messages:
            await db.flush()
