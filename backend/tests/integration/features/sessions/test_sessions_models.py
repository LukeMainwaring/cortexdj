"""Session model classmethods against the real (migrated) schema."""

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.features.sessions.model import Session
from tests.factories import create_session


class TestSession:
    async def test_list_all_paginates_newest_first(self, db_session: AsyncSession) -> None:
        for day in (1, 2, 3):
            await create_session(db_session, recorded_at=datetime(2024, 2, day, tzinfo=UTC))

        sessions, total = await Session.list_all(db_session, limit=2, offset=0)
        assert total == 3
        assert len(sessions) == 2
        assert sessions[0].recorded_at.day == 3

    async def test_get_by_participant(self, db_session: AsyncSession) -> None:
        await create_session(db_session, participant_id="P01")
        await create_session(db_session, participant_id="P02")

        found = await Session.get_by_participant(db_session, "P02")
        assert [s.participant_id for s in found] == ["P02"]
