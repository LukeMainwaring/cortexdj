"""Spotify API service using Client Credentials and User OAuth flows.

Client Credentials flow is used for public search operations.
User OAuth flow is used for playlist creation and listening history.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.core.config import get_settings

config = get_settings()

logger = logging.getLogger(__name__)

_spotify_client: spotipy.Spotify | None = None

T = TypeVar("T")

SPOTIFY_SCOPES = (
    "user-library-read playlist-read-private playlist-modify-private playlist-modify-public user-read-recently-played"
)


def get_spotify_client() -> spotipy.Spotify | None:
    """Returns None if Spotify credentials are not configured."""
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


async def get_user_spotify_client(db: AsyncSession) -> spotipy.Spotify | None:
    """Get a user-authenticated Spotify client from stored OAuth tokens.

    Auto-refreshes if token is expired or expires within 5 minutes.
    Returns None if no tokens are stored (user hasn't connected Spotify).
    """
    from cortexdj.models.spotify_token import SpotifyToken

    token = await SpotifyToken.get(db)
    if token is None:
        return None

    now = datetime.now(UTC)
    buffer = timedelta(minutes=5)
    needs_refresh = token.expires_at <= now + buffer

    access_token = token.access_token

    if needs_refresh and token.refresh_token:
        logger.info("Refreshing Spotify access token")
        try:
            oauth = SpotifyOAuth(
                client_id=config.SPOTIFY_CLIENT_ID,
                client_secret=config.SPOTIFY_CLIENT_SECRET,
                redirect_uri=config.SPOTIFY_REDIRECT_URI,
                scope=SPOTIFY_SCOPES,
            )
            token_info = await run_spotify(oauth.refresh_access_token, token.refresh_token)

            if token_info:
                expires_at = datetime.fromtimestamp(token_info["expires_at"], tz=UTC)
                await SpotifyToken.upsert(
                    db,
                    access_token=token_info["access_token"],
                    refresh_token=token_info.get("refresh_token", token.refresh_token),
                    expires_at=expires_at,
                )
                access_token = token_info["access_token"]
                logger.info("Successfully refreshed Spotify access token")
        except Exception:
            logger.warning("Failed to refresh Spotify token, continuing with existing token")

    return spotipy.Spotify(auth=access_token)


async def fetch_all_pages(
    fetch_func: Callable[..., dict[str, Any]],
    limit: int = 50,
    max_items: int | None = 500,
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], int]:
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


async def _search_paginated(
    client: spotipy.Spotify,
    query: str,
    search_type: str,
    max_results: int,
    batch_size: int = 50,
) -> tuple[list[dict[str, Any]], int]:
    result_key = f"{search_type}s"
    all_items: list[dict[str, Any]] = []
    current_offset = 0
    total_available = 0

    while len(all_items) < max_results:
        fetch_limit = min(batch_size, max_results - len(all_items))
        results = await run_spotify(client.search, q=query, type=search_type, limit=fetch_limit, offset=current_offset)
        data = results.get(result_key, {})
        items = data.get("items", [])
        total_available = data.get("total", 0)

        if not items:
            break

        all_items.extend(items)
        current_offset += len(items)

        if current_offset >= total_available:
            break

    return all_items, total_available


async def search_tracks(query: str, *, max_results: int = 10) -> list[dict[str, Any]]:
    client = get_spotify_client()
    if client is None:
        return []

    max_results = max(1, min(200, max_results))
    all_tracks, _ = await _search_paginated(client, query, "track", max_results)

    return [
        {
            "name": t["name"],
            "artists": [a["name"] for a in t["artists"]],
            "album": t["album"]["name"],
            "spotify_url": t["external_urls"].get("spotify"),
            "preview_url": t.get("preview_url"),
            "track_id": t["id"],
        }
        for t in all_tracks
    ]


async def create_playlist(
    name: str,
    track_ids: list[str],
    *,
    description: str = "",
    client: spotipy.Spotify | None = None,
) -> dict[str, Any]:
    """When a user-authenticated client is provided, creates a real playlist
    on the user's Spotify account. Otherwise returns a local-only record.
    """
    if client is None:
        logger.info(f"Would create Spotify playlist '{name}' with {len(track_ids)} tracks")
        return {
            "name": name,
            "track_count": len(track_ids),
            "description": description,
            "spotify_url": None,
            "spotify_playlist_id": None,
            "note": "Connect Spotify in Settings to create real playlists.",
        }

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
        track_uris = [f"spotify:track:{tid}" for tid in track_ids if tid]
        for i in range(0, len(track_uris), 100):
            batch = track_uris[i : i + 100]
            await run_spotify(client.playlist_add_items, playlist["id"], batch)

        logger.info(f"Created Spotify playlist '{name}' with {len(track_ids)} tracks")
        return {
            "name": name,
            "track_count": len(track_ids),
            "description": description,
            "spotify_url": playlist["external_urls"].get("spotify"),
            "spotify_playlist_id": playlist["id"],
        }
    except Exception:
        logger.exception(f"Failed to create Spotify playlist '{name}'")
        return {
            "name": name,
            "track_count": len(track_ids),
            "description": description,
            "spotify_url": None,
            "spotify_playlist_id": None,
            "note": "Failed to create playlist on Spotify. Try reconnecting in Settings.",
        }
