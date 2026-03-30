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
            logger.warning("Failed to refresh Spotify token, falling back to Client Credentials")
            return None

    return spotipy.Spotify(auth=access_token)


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
    client: spotipy.Spotify | None = None,
) -> dict[str, Any]:
    """Create a Spotify playlist and add tracks.

    When a user-authenticated client is provided, creates a real playlist
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
        # Create the playlist on Spotify
        user_info = await run_spotify(client.current_user)
        user_id = user_info["id"]

        playlist = await run_spotify(
            client.user_playlist_create,
            user_id,
            name,
            public=False,
            description=description,
        )

        # Add tracks in batches of 100 (Spotify API limit)
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
