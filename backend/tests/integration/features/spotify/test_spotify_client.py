"""The Spotify client getter: the override beats the database."""

from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.features.spotify.client import get_spotify_client, override_spotify_client
from tests.factories import create_spotify_token


async def test_the_override_wins_over_a_stored_login(db_session: AsyncSession) -> None:
    await create_spotify_token(db_session)

    with override_spotify_client(None):
        assert await get_spotify_client(db_session) is None

    assert await get_spotify_client(db_session) is not None
