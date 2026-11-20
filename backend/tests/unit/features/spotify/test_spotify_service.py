"""The Spotify service paths that need no Spotify client: unconfigured search and a local-only playlist."""

import pytest

from cortexdj.features.spotify import service as spotify_service
from cortexdj.shared.track_access import TrackInfo


async def test_search_without_a_client_finds_nothing() -> None:
    assert await spotify_service.search_tracks(None, "Queen") == []


async def test_a_track_with_a_spotify_id_resolves_to_it_without_a_search() -> None:
    track = TrackInfo(id="t-1", title="Weightless", artist="Marconi Union", spotify_track_id="sp-1")

    assert await spotify_service.resolve_track_spotify_id(None, track) == "sp-1"


async def test_a_track_without_an_id_and_no_client_resolves_to_none() -> None:
    track = TrackInfo(id="t-1", title="Weightless", artist="Marconi Union", spotify_track_id=None)

    assert await spotify_service.resolve_track_spotify_id(None, track) is None


async def test_a_playlist_without_a_client_is_a_local_record() -> None:
    record = await spotify_service.create_playlist("Calm", ["sp-1", "sp-2"], description="d")

    assert record == {
        "name": "Calm",
        "track_count": 2,
        "description": "d",
        "spotify_url": None,
        "spotify_playlist_id": None,
        "note": "Connect Spotify in Settings to create real playlists.",
    }


async def test_a_playlist_needs_tracks() -> None:
    with pytest.raises(ValueError, match="no track IDs"):
        await spotify_service.create_playlist("Calm", [])
