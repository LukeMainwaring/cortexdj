"""The session-lookup seam: other slices read a session's participant id through it."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.shared.session_lookup import (
    SessionLookupNotRegistered,
    get_session_participant_id,
    override_session_lookup,
)
from tests.factories import make_deps


async def _lookup(db: AsyncSession, session_id: str) -> str | None:
    return "P07" if session_id == "s-1" else None


async def _other_lookup(db: AsyncSession, session_id: str) -> str | None:
    return "P99"


async def test_reads_through_the_installed_lookup() -> None:
    db = make_deps().db

    with override_session_lookup(_lookup):
        assert await get_session_participant_id(db, "s-1") == "P07"
        assert await get_session_participant_id(db, "unknown") is None


async def test_override_puts_the_previous_lookup_back() -> None:
    db = make_deps().db

    with override_session_lookup(_lookup):
        with override_session_lookup(_other_lookup):
            assert await get_session_participant_id(db, "s-1") == "P99"
        assert await get_session_participant_id(db, "s-1") == "P07"


async def test_an_unregistered_lookup_is_a_wiring_error() -> None:
    with override_session_lookup(None), pytest.raises(SessionLookupNotRegistered, match="cortexdj.api"):
        await get_session_participant_id(make_deps().db, "s-1")
