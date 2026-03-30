"""Spotify API service using spotipy with Client Credentials flow.

Since CortexDJ doesn't have user auth yet, this uses client credentials
for public search and a stored access token for playlist operations.
User OAuth is a roadmap item.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, TypeVar

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from cortexdj.core.config import get_settings

config = get_settings()

logger = logging.getLogger(__name__)

_spotify_client: spotipy.Spotify | None = None

T = TypeVar("T")

SPOTIFY_SCOPES = "user-library-read playlist-read-private playlist-modify-private playlist-modify-public"


def get_spotify_client() -> spotipy.Spotify | None:
    """Get or create singleton Spotify client (Client Credentials flow).

    Returns None if Spotify credentials are not configured.
    """
    global _spotify_client
    if _spotify_client is None and config.spotify_configured:
        _spotify_client = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=config.SPOTIFY_CLIENT_ID,
                client_secret=config.SPOTIFY_CLIENT_SECRET,
            )
        )
    return _spotify_client


async def run_spotify(func: Callable[..., T], *args: object, **kwargs: object) -> T:
    """Wrap synchronous spotipy call in asyncio.to_thread for async compatibility."""
    return await asyncio.to_thread(func, *args, **kwargs)


async def fetch_all_pages(
    fetch_func: Callable[..., dict[str, Any]],
    limit: int = 50,
    max_items: int | None = 500,
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], int]:
    """Fetch all pages from a paginated Spotify endpoint."""
    all_items: list[dict[str, Any]] = []
    offset = 0
    total = None

    while True:
        if max_items is not None and len(all_items) >= max_items:
            break
        results = await run_spotify(fetch_func, limit=limit, offset=offset, **kwargs)
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


async def search_tracks(query: str, *, limit: int = 10) -> list[dict[str, Any]]:
    """Search Spotify for tracks by query string."""
    client = get_spotify_client()
    if client is None:
        return []

    results = await run_spotify(client.search, q=query, type="track", limit=limit)
    tracks_data = results.get("tracks", {}).get("items", [])

    return [
        {
            "name": t["name"],
            "artists": [a["name"] for a in t["artists"]],
            "album": t["album"]["name"],
            "spotify_url": t["external_urls"].get("spotify"),
            "track_id": t["id"],
        }
        for t in tracks_data
    ]


async def create_playlist(
    name: str,
    track_ids: list[str],
    *,
    description: str = "",
) -> dict[str, Any] | None:
    """Create a Spotify playlist and add tracks.

    Note: This requires user OAuth (roadmap item). Currently returns
    a local-only playlist record.
    """
    # TODO: Implement with user OAuth when auth is added
    logger.info(f"Would create Spotify playlist '{name}' with {len(track_ids)} tracks")
    return {
        "name": name,
        "track_count": len(track_ids),
        "description": description,
        "spotify_url": None,
        "note": "Spotify OAuth required for real playlist creation (see ROADMAP)",
    }
