"""The Spotify client override: what a test sees inside it, and what comes back after it."""

from cortexdj.features.spotify.client import SpotifyClient, get_spotify_client, override_spotify_client
from tests.factories import make_deps


async def test_nested_overrides_restore_the_previous_client_on_exit() -> None:
    outer = SpotifyClient(client=None, is_user=False)
    inner = SpotifyClient(client=None, is_user=True)
    db = make_deps().db

    with override_spotify_client(outer):
        with override_spotify_client(inner):
            assert await get_spotify_client(db) is inner
        assert await get_spotify_client(db) is outer
