"""Notes agent tools against the real service and database, called the way the agent calls them."""

from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.features.notes import tools as note_tools
from tests.factories import create_note, make_deps, make_run_context


async def test_list_notes_tool_returns_stored_notes_and_total(db_session: AsyncSession) -> None:
    await create_note(db_session, title="Milk", content="2%")

    result = await note_tools.list_notes(make_run_context(make_deps(db_session)))

    assert result["total"] == 1
    assert result["notes"][0]["title"] == "Milk"
    assert result["notes"][0]["content"] == "2%"


async def test_create_note_tool_saves_a_note_the_list_tool_returns(db_session: AsyncSession) -> None:
    ctx = make_run_context(make_deps(db_session))

    created = await note_tools.create_note(ctx, title="Eggs", content="dozen")
    listed = await note_tools.list_notes(ctx)

    assert created["title"] == "Eggs"
    assert [note["id"] for note in listed["notes"]] == [created["id"]]
