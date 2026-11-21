"""Chat model classmethods against the real schema, for what no endpoint can show."""

from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.features.chat.model import Message
from tests.factories import create_thread


async def test_append_messages_with_nothing_new_is_a_no_op(db_session: AsyncSession) -> None:
    await create_thread(db_session, "t-empty")

    await Message.append_messages(db_session, "t-empty", [])

    assert await Message.get_history(db_session, "t-empty") == []
