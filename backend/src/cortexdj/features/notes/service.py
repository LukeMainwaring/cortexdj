from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.features.notes.model import Note


async def list_notes(db: AsyncSession, *, limit: int = 50, offset: int = 0) -> tuple[list[Note], int]:
    return await Note.list_all(db, limit=limit, offset=offset)


async def create_note(db: AsyncSession, *, title: str, content: str) -> Note:
    note = await Note.create(db, title=title, content=content)
    return note


async def get_note(db: AsyncSession, note_id: str) -> Note | None:
    return await Note.get(db, note_id)
