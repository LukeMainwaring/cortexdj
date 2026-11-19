from fastapi import APIRouter, HTTPException, status

from cortexdj.core.db import AsyncPostgresSessionDep
from cortexdj.features.notes import service as note_service
from cortexdj.features.notes.schema import NoteCreate, NoteListResponse, NoteSchema

notes_router = APIRouter(prefix="/notes", tags=["notes"])


@notes_router.get("")
async def list_notes(db: AsyncPostgresSessionDep) -> NoteListResponse:
    """List the user's notes, newest first."""
    notes, total = await note_service.list_notes(db)
    return NoteListResponse(notes=[NoteSchema.model_validate(n) for n in notes], total=total)


@notes_router.post("", status_code=status.HTTP_201_CREATED)
async def create_note(db: AsyncPostgresSessionDep, body: NoteCreate) -> NoteSchema:
    """Create a new note."""
    note = await note_service.create_note(db, title=body.title, content=body.content)
    return NoteSchema.model_validate(note)


@notes_router.get("/{note_id}")
async def get_note(db: AsyncPostgresSessionDep, note_id: str) -> NoteSchema:
    """Get a single note by id."""
    note = await note_service.get_note(db, note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    return NoteSchema.model_validate(note)
