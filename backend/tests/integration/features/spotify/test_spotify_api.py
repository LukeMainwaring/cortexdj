"""Route-level tests for the Spotify connection status, asserted through HTTP."""

from datetime import UTC, datetime, timedelta

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from tests.factories import create_spotify_token


async def test_status_is_not_connected_without_a_login(client: AsyncClient) -> None:
    response = await client.get("/api/spotify/status")

    assert response.status_code == 200
    assert response.json() == {"connected": False}


async def test_status_is_connected_with_a_stored_login(client: AsyncClient, db_session: AsyncSession) -> None:
    await create_spotify_token(db_session)

    assert (await client.get("/api/spotify/status")).json() == {"connected": True}


async def test_status_is_not_connected_when_the_stored_login_cannot_refresh(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    # Expired, and unconfigured settings make the refresh fail before any request is sent.
    await create_spotify_token(db_session, expires_at=datetime.now(UTC) - timedelta(minutes=1))

    assert (await client.get("/api/spotify/status")).json() == {"connected": False}
