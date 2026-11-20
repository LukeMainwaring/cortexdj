"""The Spotify token singleton against the real (migrated) schema."""

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.features.spotify.model import SpotifyToken


class TestSpotifyToken:
    async def test_upsert_then_update_keeps_singleton(self, db_session: AsyncSession) -> None:
        expires = datetime(2030, 1, 1, tzinfo=UTC)
        assert not await SpotifyToken.is_connected(db_session)

        await SpotifyToken.upsert(db_session, "access-1", "refresh-1", expires)
        token = await SpotifyToken.upsert(db_session, "access-2", "refresh-2", expires)

        assert token.access_token == "access-2"
        assert await SpotifyToken.is_connected(db_session)

    async def test_clear(self, db_session: AsyncSession) -> None:
        await SpotifyToken.upsert(db_session, "a", "r", datetime(2030, 1, 1, tzinfo=UTC))
        await SpotifyToken.clear(db_session)
        assert not await SpotifyToken.is_connected(db_session)
