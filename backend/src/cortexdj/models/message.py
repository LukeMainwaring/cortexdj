from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKeyConstraint, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from cortexdj.models.base import Base


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    thread_id: Mapped[str]
    agent_type: Mapped[str]
    message_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now)

    __table_args__ = (
        ForeignKeyConstraint(
            ["thread_id", "agent_type"],
            ["threads.thread_id", "threads.agent_type"],
            ondelete="CASCADE",
        ),
    )

    @classmethod
    async def get_history(cls, db: AsyncSession, thread_id: str, agent_type: str) -> list[dict[str, Any]]:
        result = await db.execute(
            select(cls.message_data)
            .where(cls.thread_id == thread_id, cls.agent_type == agent_type)
            .order_by(cls.id.asc())
        )
        return [row[0] for row in result.all()]

    @classmethod
    async def save_history(
        cls, db: AsyncSession, thread_id: str, agent_type: str, messages: list[dict[str, Any]]
    ) -> None:
        """Append new messages to thread history.

        Uses an append-only approach that only inserts messages beyond the
        existing count, preserving original timestamps and IDs.
        """
        result = await db.execute(
            select(func.count(cls.id)).where(cls.thread_id == thread_id, cls.agent_type == agent_type)
        )
        existing_count = result.scalar() or 0
        new_messages = messages[existing_count:]
        if new_messages:
            for msg_data in new_messages:
                message = cls(
                    thread_id=thread_id,
                    agent_type=agent_type,
                    message_data=msg_data,
                )
                db.add(message)
            await db.flush()
