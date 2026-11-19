"""Thread lifecycle through the HTTP interface: list/rename/delete/404 + cascade."""

import asyncio

from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from cortexdj.features.chat.model import Message, Thread
from tests.factories import create_thread


async def test_get_messages_unknown_thread_returns_404(client: AsyncClient) -> None:
    resp = await client.get("/api/threads/nope/messages")
    assert resp.status_code == 404


async def test_rename_unknown_thread_returns_404(client: AsyncClient) -> None:
    resp = await client.patch("/api/threads/nope", json={"title": "x"})
    assert resp.status_code == 404


async def test_delete_unknown_thread_returns_404(client: AsyncClient) -> None:
    resp = await client.delete("/api/threads/nope")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Thread not found"}


async def test_a_stored_thread_is_listed_renamable_and_gone_once_deleted(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    await create_thread(db_session, thread_id="t-1")

    listed = await client.get("/api/threads")
    assert listed.status_code == 200
    assert [t["thread_id"] for t in listed.json()["threads"]] == ["t-1"]

    renamed = await client.patch("/api/threads/t-1", json={"title": "Renamed"})
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Renamed"

    deleted = await client.delete("/api/threads/t-1")
    assert deleted.status_code == 200

    after = await client.get("/api/threads/t-1/messages")
    assert after.status_code == 404


async def test_delete_thread_cascades_to_messages(client: AsyncClient, db_session: AsyncSession) -> None:
    await create_thread(db_session, thread_id="t-2")
    await Message.append_messages(db_session, "t-2", [{"k": "v1"}, {"k": "v2"}])
    assert len(await Message.get_history(db_session, "t-2")) == 2

    deleted = await client.delete("/api/threads/t-2")
    assert deleted.status_code == 200

    # FK ondelete=CASCADE removed the messages along with the thread. Read at the model layer:
    # the messages route 404s once the thread is gone, so no endpoint can see them.
    assert await Message.get_history(db_session, "t-2") == []


async def _until_a_session_waits_on_a_lock(db: AsyncSession) -> bool:
    """True once another backend is blocked on a lock, so the test exercises the duplicate-key path."""
    # A deadline checked between polls, not `asyncio.timeout`: cancelling `db` mid-statement
    # would break its connection, and the test's commit and cleanup would fail on that instead.
    deadline = asyncio.get_running_loop().time() + 5.0
    while asyncio.get_running_loop().time() < deadline:
        # pg_stat_activity is snapshotted once per transaction, and `db` is mid-transaction
        # (its insert is uncommitted), so each poll must drop the snapshot first.
        await db.execute(text("SELECT pg_stat_clear_snapshot()"))
        waiting = await db.scalar(text("SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock'"))
        if waiting:
            return True
        await asyncio.sleep(0.02)
    return False


async def test_two_first_turns_on_one_new_thread_id_both_get_the_same_thread(engine: AsyncEngine) -> None:
    """The loser of the insert race gets the winner's row, not a 500.

    Two real connections, outside the per-test rollback: the second insert must block on the
    first's uncommitted row and then fail as a duplicate, which one transaction cannot stage.
    The row is deleted and committed at the end, so the database is left as found. The loser
    task is always awaited or cancelled before its session is touched again: rolling back a
    session whose connection is mid-statement in another task waits forever.
    """
    sessions = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with sessions() as first, sessions() as second:
        await Thread.get_or_create(first, "t-race")  # uncommitted: invisible to `second`
        loser = asyncio.create_task(Thread.get_or_create(second, "t-race"))
        try:
            blocked = await _until_a_session_waits_on_a_lock(first)
            await first.commit()  # releases the row lock either way
            thread = await asyncio.wait_for(loser, timeout=10)
            assert blocked, "the second insert never blocked on the first's row lock"
            assert thread.thread_id == "t-race"
            assert await Thread.get(second, "t-race") is thread
        finally:
            if not loser.done():
                loser.cancel()
                await asyncio.gather(loser, return_exceptions=True)
            await second.rollback()
            await Thread.delete_by_id(first, "t-race")
            await first.commit()
