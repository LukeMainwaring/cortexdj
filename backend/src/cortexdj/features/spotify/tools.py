import asyncio
import logging
import uuid
from typing import Any

import spotipy
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ToolFailed

from cortexdj.agents.deps import AgentDeps
from cortexdj.features.spotify import service as spotify_service
from cortexdj.features.spotify.client import get_spotify_client
from cortexdj.features.spotify.model import Playlist
from cortexdj.shared import track_access
from cortexdj.shared.track_access import TrackInfo

_RESOLVE_CONCURRENCY = 5

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Failures the model sees (raised as ToolFailed; see agents/hooks.py)
# ---------------------------------------------------------------------------


def _requires_connection() -> ToolFailed:
    return ToolFailed("Spotify is not connected. Go to Settings > Spotify to connect.")


def _token_expired() -> ToolFailed:
    return ToolFailed("Your Spotify connection has expired. Please reconnect in Settings > Spotify.")


def _spotify_api_error(operation: str, error: Exception) -> ToolFailed:
    return ToolFailed(f"Failed to {operation}: {error!s}")


async def _require_client(ctx: RunContext[AgentDeps]) -> spotipy.Spotify:
    """The user's client, or ToolFailed when Spotify is not connected.

    The capability hides every tool here without a login, so this fails only when the
    user disconnects, or a refresh fails, between tool preparation and the call.
    """
    client = await get_spotify_client(ctx.deps.db)
    if client is None:
        raise _requires_connection()
    return client


# ---------------------------------------------------------------------------
# Mood playlists
# ---------------------------------------------------------------------------


async def build_mood_playlist(
    ctx: RunContext[AgentDeps],
    mood: str,
    name: str | None = None,
    user_confirmed: bool = False,
) -> dict[str, Any]:
    """Build a Spotify playlist from tracks matching a brain-derived mood.

    IMPORTANT: You must get explicit user confirmation before calling this tool.
    Always propose the playlist first and ask if the user wants to proceed.

    Args:
        mood: Target mood (relaxed, calm, excited, stressed)
        name: Optional playlist name (auto-generated if not provided)
        user_confirmed: Must be True to confirm user has approved creation
    """
    if not user_confirmed:
        raise ToolFailed(
            "You must get explicit user confirmation before creating a playlist. "
            "Propose the playlist name and track selection, then call this tool "
            "with user_confirmed=True after they approve."
        )

    tracks = await track_access.find_tracks_by_mood(ctx.deps.db, mood, limit=30)

    if not tracks:
        raise ToolFailed(f"No tracks found for mood '{mood}'. Try a different mood or seed more data.")

    playlist_name = name or f"CortexDJ: {mood.capitalize()} Mix"

    client = await _require_client(ctx)

    internal_ids = [t.track_id for t in tracks]
    db_tracks = await track_access.get_tracks(ctx.deps.db, internal_ids)

    semaphore = asyncio.Semaphore(_RESOLVE_CONCURRENCY)

    async def _resolve(t: TrackInfo) -> tuple[TrackInfo, str | None]:
        async with semaphore:
            return t, await spotify_service.resolve_track_spotify_id(client, t)

    resolution = await asyncio.gather(*(_resolve(t) for t in db_tracks))
    resolved_pairs: list[tuple[TrackInfo, str]] = [(t, sid) for t, sid in resolution if sid]
    dropped_count = len(db_tracks) - len(resolved_pairs)

    if dropped_count:
        logger.warning(f"Dropped {dropped_count}/{len(db_tracks)} tracks for mood '{mood}' (no Spotify match)")

    if not resolved_pairs:
        raise ToolFailed("Could not resolve any of the mood-matched tracks to Spotify. The playlist was not created.")

    # Persist newly-resolved IDs serially (AsyncSession is not safe for concurrent writes).
    # save_track_spotify_id skips a write that would collide with a track already holding
    # that Spotify ID; the playlist can still include the track.
    for t, sid in resolved_pairs:
        if t.spotify_track_id == sid:
            continue
        await track_access.save_track_spotify_id(ctx.deps.db, t.id, sid)

    spotify_track_ids = [sid for _, sid in resolved_pairs]

    playlist = Playlist(
        id=str(uuid.uuid4()),
        name=playlist_name,
        mood_criteria={"target_mood": mood, "track_count": len(resolved_pairs)},
        track_count=len(resolved_pairs),
    )
    ctx.deps.db.add(playlist)
    await ctx.deps.db.flush()

    spotify_result = await spotify_service.create_playlist(
        playlist_name,
        spotify_track_ids,
        client=client,
        description=f"Auto-curated by CortexDJ based on brain state: {mood}",
    )

    if spotify_result.get("spotify_playlist_id"):
        playlist.spotify_playlist_id = spotify_result["spotify_playlist_id"]
        await ctx.deps.db.flush()

    # A failure on Spotify's side is not raised: the playlist is saved in the database and
    # its tracks are the result, and ``note`` tells the model the copy on Spotify failed.
    return {
        "playlist_name": playlist_name,
        "mood": mood,
        "track_count": len(resolved_pairs),
        "tracks": [{"title": t.title, "artist": t.artist} for t, _ in resolved_pairs[:15]],
        "tracks_truncated": len(resolved_pairs) > 15,
        "dropped_track_count": dropped_count,
        "spotify_url": spotify_result.get("spotify_url"),
        "spotify_playlist_id": spotify_result.get("spotify_playlist_id"),
        "note": spotify_result.get("note"),
    }


# ---------------------------------------------------------------------------
# The user's Spotify library
# ---------------------------------------------------------------------------


async def get_listening_history(ctx: RunContext[AgentDeps], limit: int = 20) -> dict[str, Any]:
    """Get recent Spotify listening history.

    Returns recently played tracks
    which can be correlated with EEG session data.
    """
    client = await _require_client(ctx)

    try:
        results = await spotify_service.run_spotify(client.current_user_recently_played, limit=limit)
    except spotipy.SpotifyException as e:
        if e.http_status == 401:
            raise _token_expired() from e
        raise _spotify_api_error("get listening history", e) from e

    items = results.get("items", [])
    if not items:
        return {"tracks": [], "count": 0, "message": "No recent listening history found."}

    return {
        "tracks": [
            {
                "name": item["track"]["name"],
                "artists": [a["name"] for a in item["track"]["artists"]],
                "played_at": item.get("played_at", ""),
                "track_id": item["track"]["id"],
            }
            for item in items
        ],
        "count": len(items),
    }


async def get_my_playlists(
    ctx: RunContext[AgentDeps],
    max_results: int | None = 50,
) -> dict[str, Any]:
    """Get the user's Spotify playlists.

    Useful for finding existing playlists
    or correlating with EEG session data.
    """
    client = await _require_client(ctx)

    try:
        items, total = await spotify_service.fetch_all_pages(
            client.current_user_playlists,
            limit=50,
            max_items=max_results,
        )
    except spotipy.SpotifyException as e:
        if e.http_status == 401:
            raise _token_expired() from e
        raise _spotify_api_error("get playlists", e) from e

    response: dict[str, Any] = {
        "playlists": [
            {
                "name": playlist["name"],
                "track_count": playlist["tracks"]["total"],
                "playlist_id": playlist["id"],
            }
            for playlist in items
        ],
        "total": total,
        "fetched_count": len(items),
    }
    if max_results is not None and len(items) >= max_results and len(items) < total:
        response["capped"] = True
    return response


async def add_tracks_to_playlist(
    ctx: RunContext[AgentDeps],
    playlist_id: str,
    track_ids: list[str],
    user_confirmed: bool = False,
) -> dict[str, Any]:
    """Add tracks to an existing Spotify playlist.

    IMPORTANT: You must get explicit user confirmation before calling this tool.
    Tell the user which tracks you plan to add and to which playlist, then call
    with user_confirmed=True after they approve.

    Args:
        playlist_id: The Spotify playlist ID to add tracks to
        track_ids: List of Spotify track IDs to add (max 100 per call)
        user_confirmed: Must be True to confirm user has approved the addition
    """
    if not user_confirmed:
        raise ToolFailed(
            "You must get explicit user confirmation before adding tracks to a playlist. "
            "Tell the user which tracks you plan to add and to which playlist, then call "
            "with user_confirmed=True after they approve."
        )

    client = await _require_client(ctx)

    if not track_ids:
        raise ToolFailed("No track IDs provided. Please specify at least one track to add.")

    if len(track_ids) > 100:
        raise ToolFailed(
            f"Too many tracks ({len(track_ids)}). Maximum 100 tracks per call. "
            "Split into multiple calls for larger additions."
        )

    try:
        track_uris = [f"spotify:track:{track_id}" for track_id in track_ids]
        await spotify_service.run_spotify(client.playlist_add_items, playlist_id, track_uris)
    except spotipy.SpotifyException as e:
        if e.http_status == 404:
            raise ToolFailed(f"Playlist not found: {playlist_id}") from e
        if e.http_status == 403:
            raise ToolFailed(
                "You don't have permission to modify this playlist. You can only add tracks to playlists you own."
            ) from e
        if e.http_status == 401:
            raise _token_expired() from e
        raise _spotify_api_error("add tracks", e) from e

    return {
        "tracks_added": len(track_ids),
        "playlist_id": playlist_id,
    }


# ---------------------------------------------------------------------------
# Catalog search
# ---------------------------------------------------------------------------


async def search_tracks(
    ctx: RunContext[AgentDeps],
    query: str,
    max_results: int = 10,
) -> dict[str, Any]:
    """Search for tracks on Spotify by name, artist, or keywords.

    Args:
        query: Search query (e.g., "Bohemian Rhapsody", "Queen", "chill electronic")
        max_results: Total tracks to return (default 10, max 200)
    """
    client = await _require_client(ctx)

    max_results = max(1, min(200, max_results))

    try:
        all_tracks, total_available = await spotify_service.search_paginated(client, query, "track", max_results)
    except spotipy.SpotifyException as e:
        raise _spotify_api_error("search tracks", e) from e

    return {
        "tracks": [
            {
                "name": track["name"],
                "artists": [a["name"] for a in track["artists"]],
                "album": track["album"]["name"],
                "spotify_url": track["external_urls"].get("spotify"),
                "track_id": track["id"],
            }
            for track in all_tracks
        ],
        "total_results": total_available,
        "fetched_count": len(all_tracks),
    }
