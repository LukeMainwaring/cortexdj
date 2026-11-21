"""Spotify Web API calls the slice's tools and auth routes make, on a client the caller passes in.

The client is always the user's own login, from ``client.get_spotify_client``; a caller
with no client does not call this module. spotipy is synchronous, so every call goes
through ``run_spotify`` to keep it off the event loop.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import spotipy

from cortexdj.shared.track_access import TrackInfo

logger = logging.getLogger(__name__)


SPOTIFY_SCOPES = (
    "user-library-read playlist-read-private playlist-modify-private playlist-modify-public user-read-recently-played"
)


async def run_spotify[T](func: Callable[..., T], *args: object, **kwargs: object) -> T:
    """Wrap synchronous spotipy call in asyncio.to_thread for async compatibility."""
    return await asyncio.to_thread(func, *args, **kwargs)


async def fetch_all_pages(
    fetch_func: Callable[..., dict[str, Any]],
    max_items: int | None = 500,
) -> tuple[list[dict[str, Any]], int]:
    all_items: list[dict[str, Any]] = []
    offset = 0
    total = None

    while True:
        if max_items is not None and len(all_items) >= max_items:
            break
        results = await run_spotify(fetch_func, limit=50, offset=offset)
        items = results.get("items", [])
        total = results.get("total", 0)

        if not items:
            break

        all_items.extend(items)
        offset += len(items)
        if offset >= total:
            break

    if max_items is not None:
        all_items = all_items[:max_items]

    return all_items, total or 0


async def search_paginated(
    client: spotipy.Spotify,
    query: str,
    max_results: int,
) -> tuple[list[dict[str, Any]], int]:
    all_items: list[dict[str, Any]] = []
    current_offset = 0
    total_available = 0

    while len(all_items) < max_results:
        fetch_limit = min(50, max_results - len(all_items))
        results = await run_spotify(client.search, q=query, type="track", limit=fetch_limit, offset=current_offset)
        data = results.get("tracks", {})
        items = data.get("items", [])
        total_available = data.get("total", 0)

        if not items:
            break

        all_items.extend(items)
        current_offset += len(items)

        if current_offset >= total_available:
            break

    return all_items, total_available


async def search_tracks(
    client: spotipy.Spotify, query: str, *, max_results: int = 10
) -> tuple[list[dict[str, Any]], int]:
    """Up to ``max_results`` tracks (clamped to 1-200) and how many Spotify has for the query."""
    max_results = max(1, min(200, max_results))
    all_tracks, total_available = await search_paginated(client, query, max_results)

    tracks = [
        {
            "name": t["name"],
            "artists": [a["name"] for a in t["artists"]],
            "album": t["album"]["name"],
            "spotify_url": t["external_urls"].get("spotify"),
            "track_id": t["id"],
        }
        for t in all_tracks
    ]
    return tracks, total_available


async def resolve_track_spotify_id(client: spotipy.Spotify, track: TrackInfo) -> str | None:
    """Return the Spotify track ID for `track`, looking it up if missing.

    Pure lookup — does NOT touch the DB. Callers persist the returned ID serially through
    ``track_access.save_track_spotify_id``, because this function is typically run
    concurrently via `asyncio.gather` and `AsyncSession` is not safe for concurrent use.

    Returns None if no match is found or the search fails. A login Spotify rejects (401)
    is raised instead: it fails every lookup the same way, and the caller tells the user
    to reconnect.
    """
    if track.spotify_track_id:
        return track.spotify_track_id

    # Strip quotes so they can't close a field or break the Spotify query grammar.
    title = track.title.replace('"', "")
    artist = track.artist.replace('"', "")
    query = f'track:"{title}" artist:"{artist}"'

    try:
        results, _ = await search_tracks(client, query, max_results=1)
    except Exception as e:  # noqa: BLE001 — one bad lookup shouldn't fail the whole playlist
        if isinstance(e, spotipy.SpotifyException) and e.http_status == 401:
            raise
        logger.warning(f"Spotify search failed for '{track.title}' by '{track.artist}': {e}")
        return None

    if not results:
        logger.warning(f"No Spotify match for '{track.title}' by '{track.artist}'")
        return None

    track_id: str = results[0]["track_id"]
    return track_id


async def create_playlist(
    name: str,
    track_ids: list[str],
    *,
    client: spotipy.Spotify,
    description: str = "",
) -> dict[str, Any]:
    """Create a private playlist on the user's Spotify account and add the tracks.

    A failure on Spotify's side is logged and comes back as a record with no
    ``spotify_playlist_id`` and a ``note``, not an exception: the caller has already
    saved the playlist locally and still reports it.
    """
    try:
        user_info = await run_spotify(client.current_user)
        user_id = user_info["id"]

        playlist = await run_spotify(
            client.user_playlist_create,
            user_id,
            name,
            public=False,
            description=description,
        )

        # Spotify API limits playlist_add_items to 100 tracks per call
        track_uris = [f"spotify:track:{tid}" for tid in track_ids]
        for i in range(0, len(track_uris), 100):
            batch = track_uris[i : i + 100]
            await run_spotify(client.playlist_add_items, playlist["id"], batch)

        logger.info(f"Created Spotify playlist '{name}' with {len(track_ids)} tracks")
        return {
            "spotify_url": playlist["external_urls"].get("spotify"),
            "spotify_playlist_id": playlist["id"],
        }
    except Exception:
        logger.exception(f"Failed to create Spotify playlist '{name}'")
        return {
            "spotify_url": None,
            "spotify_playlist_id": None,
            "note": "Failed to create playlist on Spotify. Try reconnecting in Settings.",
        }
