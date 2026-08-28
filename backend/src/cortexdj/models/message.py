from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKeyConstraint, select
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
    async def append_messages(
        cls, db: AsyncSession, thread_id: str, agent_type: str, messages: list[dict[str, Any]]
    ) -> None:
        """Append messages to the thread history; the caller decides what is new.

        Deliberately dumb: it inserts exactly what it is given. The predecessor
        inferred "new" from the stored row count, which only held while a single
        request wrote the whole conversation at once. ``routers/agent.py`` now
        writes the user's turn before the run and the run's own messages after
        it, so the count is stale by the time the second write happens.
        """
        for msg_data in messages:
            db.add(cls(thread_id=thread_id, agent_type=agent_type, message_data=msg_data))
        if messages:
            await db.flush()
