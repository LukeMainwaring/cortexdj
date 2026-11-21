"""Resolving a track's Spotify id without reaching Spotify: a track that already has its id, and a failed search."""

from typing import Any

from cortexdj.features.spotify import service as spotify_service
from cortexdj.shared.track_access import TrackInfo

# Stands for a client that must not be called: any spotipy call on it raises AttributeError.
UNUSED_CLIENT: Any = object()


async def test_a_track_with_a_spotify_id_resolves_to_it_without_a_search() -> None:
    track = TrackInfo(id="t-1", title="Weightless", artist="Marconi Union", spotify_track_id="sp-1")

    assert await spotify_service.resolve_track_spotify_id(UNUSED_CLIENT, track) == "sp-1"


async def test_a_track_whose_search_fails_resolves_to_none() -> None:
    track = TrackInfo(id="t-1", title="Weightless", artist="Marconi Union", spotify_track_id=None)

    assert await spotify_service.resolve_track_spotify_id(UNUSED_CLIENT, track) is None
