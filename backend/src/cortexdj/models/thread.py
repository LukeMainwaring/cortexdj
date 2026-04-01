from __future__ import annotations

from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import DateTime, PrimaryKeyConstraint, delete, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from cortexdj.models.base import Base
from cortexdj.schemas.agent_type import AgentType
from cortexdj.schemas.thread import BrainContext, ThreadCreateSchema, ThreadSchema


class ThreadNotFound(HTTPException):
    def __init__(self, detail: str = "Thread not found"):
        super().__init__(status_code=404, detail=detail)


class Thread(Base):
    __tablename__ = "threads"

    thread_id: Mapped[str]
    agent_type: Mapped[str]
    title: Mapped[str | None]
    brain_context: Mapped[dict[str, object] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now, onupdate=datetime.now)

    __table_args__ = (PrimaryKeyConstraint("thread_id", "agent_type"),)

    @classmethod
    async def get(cls, db: AsyncSession, thread_id: str, agent_type: str) -> Thread | None:
        result = await db.execute(select(cls).where(cls.thread_id == thread_id, cls.agent_type == agent_type))
        return result.scalar_one_or_none()

    @classmethod
    async def create(cls, db: AsyncSession, thread_create: ThreadCreateSchema) -> ThreadSchema:
        thread = cls(
            thread_id=thread_create.thread_id,
            agent_type=thread_create.agent_type.value,
        )
        db.add(thread)
        await db.flush()
        await db.refresh(thread)
        return ThreadSchema.model_validate(thread)

    @classmethod
    async def get_or_create(cls, db: AsyncSession, thread_id: str, agent_type: str) -> ThreadSchema:
        thread = await cls.get(db, thread_id, agent_type)
        if thread:
            return ThreadSchema.model_validate(thread)
        return await cls.create(
            db,
            ThreadCreateSchema(
                thread_id=thread_id,
                agent_type=AgentType(agent_type),
            ),
        )

    @classmethod
    async def list_all(cls, db: AsyncSession, agent_type: str) -> list[Thread]:
        result = await db.execute(select(cls).where(cls.agent_type == agent_type).order_by(cls.updated_at.desc()))
        return list(result.scalars().all())

    @classmethod
    async def delete_by_id(cls, db: AsyncSession, thread_id: str, agent_type: str) -> None:
        await db.execute(delete(cls).where(cls.thread_id == thread_id, cls.agent_type == agent_type))
        await db.flush()

    @classmethod
    async def update_title(cls, db: AsyncSession, thread_id: str, agent_type: str, title: str) -> None:
        thread = await cls.get(db, thread_id, agent_type)
        if thread:
            thread.title = title
            await db.flush()

    @classmethod
    async def update_brain_context(
        cls, db: AsyncSession, thread_id: str, agent_type: str, updates: BrainContext
    ) -> BrainContext:
        """Merge brain context updates into the thread's existing context.

        Uses exclude_unset so only explicitly provided fields are overwritten.
        Creates the thread if it doesn't exist yet (first message in a conversation).
        """
        thread = await cls.get(db, thread_id, agent_type)
        if thread is None:
            await cls.create(db, ThreadCreateSchema(thread_id=thread_id, agent_type=AgentType(agent_type)))
            thread = await cls.get(db, thread_id, agent_type)
            assert thread is not None
        existing = BrainContext.model_validate(thread.brain_context) if thread.brain_context else BrainContext()
        merged = existing.model_copy(update=updates.model_dump(exclude_unset=True))
        thread.brain_context = merged.model_dump(exclude_none=True)
        await db.flush()
        return merged
