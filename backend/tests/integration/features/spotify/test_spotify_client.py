"""The Spotify client getter: the stored login decides the client, and the override beats the database."""

from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.features.spotify.client import get_spotify_client, override_spotify_client
from tests.factories import create_spotify_token


async def test_a_stored_login_gives_the_users_client(db_session: AsyncSession) -> None:
    await create_spotify_token(db_session)

    assert await get_spotify_client(db_session) is not None


async def test_without_a_login_there_is_no_client(db_session: AsyncSession) -> None:
    assert await get_spotify_client(db_session) is None


async def test_a_login_whose_refresh_fails_counts_as_not_connected(db_session: AsyncSession) -> None:
    # Expired, and unconfigured settings make the refresh fail before any request is sent.
    await create_spotify_token(db_session, expires_at=datetime.now(UTC) - timedelta(minutes=1))

    assert await get_spotify_client(db_session) is None


async def test_the_override_wins_over_a_stored_login(db_session: AsyncSession) -> None:
    await create_spotify_token(db_session)

    with override_spotify_client(None):
        assert await get_spotify_client(db_session) is None

    assert await get_spotify_client(db_session) is not None
