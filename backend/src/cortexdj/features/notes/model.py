import uuid
from datetime import datetime

from sqlalchemy import DateTime, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from cortexdj.shared.model_base import Base


class Note(Base):
    __tablename__ = "notes"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: uuid.uuid4().hex)
    title: Mapped[str]
    content: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    @classmethod
    async def get(cls, db: AsyncSession, id: str) -> Note | None:
        result = await db.execute(select(cls).where(cls.id == id))
        return result.scalar_one_or_none()

    @classmethod
    async def list_all(cls, db: AsyncSession, *, limit: int = 50, offset: int = 0) -> tuple[list[Note], int]:
        total = await db.scalar(select(func.count()).select_from(cls)) or 0
        result = await db.execute(select(cls).order_by(cls.created_at.desc()).limit(limit).offset(offset))
        return list(result.scalars().all()), total

    @classmethod
    async def create(cls, db: AsyncSession, *, title: str, content: str) -> Note:
        note = cls(title=title, content=content)
        db.add(note)
        await db.flush()
        await db.refresh(note)
        return note
