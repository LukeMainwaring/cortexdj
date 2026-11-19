from typing import Any

from pydantic_ai import RunContext

from cortexdj.agents.deps import AgentDeps
from cortexdj.features.notes import service as note_service
from cortexdj.features.notes.schema import NoteSchema


async def list_notes(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
    """List the user's saved notes (newest first)."""
    notes, total = await note_service.list_notes(ctx.deps.db)
    return {
        "notes": [NoteSchema.model_validate(n).model_dump(mode="json") for n in notes],
        "total": total,
    }


async def create_note(ctx: RunContext[AgentDeps], title: str, content: str) -> dict[str, Any]:
    """Create a new note with a title and content, and return the saved note.

    Args:
        title: Short heading for the note.
        content: The note body, as plain text.
    """
    note = await note_service.create_note(ctx.deps.db, title=title, content=content)
    return NoteSchema.model_validate(note).model_dump(mode="json")
