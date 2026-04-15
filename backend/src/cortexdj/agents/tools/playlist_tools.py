import logging
import uuid
from typing import Any

import spotipy
from pydantic_ai import RunContext

from cortexdj.agents.deps import AgentDeps
from cortexdj.models.playlist import Playlist
from cortexdj.models.track import Track
from cortexdj.services import eeg_processing as eeg_service
from cortexdj.services.spotify import (
    create_playlist,
    fetch_all_pages,
    get_spotify_client,
    run_spotify,
    search_paginated,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error helpers (structured dicts for consistent agent parsing)
# ---------------------------------------------------------------------------


def _spotify_not_configured_error() -> dict[str, Any]:
    return {"error": "Spotify is not configured. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET."}


def _requires_connection_error() -> dict[str, Any]:
    return {
        "error": "requires_connection",
        "message": "Spotify is not connected. Go to Settings > Spotify to connect.",
    }


def _token_expired_error() -> dict[str, Any]:
    return {
        "error": "token_expired",
        "message": "Your Spotify connection has expired. Please reconnect in Settings > Spotify.",
    }


def _spotify_api_error(operation: str, error: Exception) -> dict[str, Any]:
    return {"error": f"Failed to {operation}: {error!s}"}


# ---------------------------------------------------------------------------
# EEG-based tools (always available)
# ---------------------------------------------------------------------------


async def find_relaxing_tracks(ctx: RunContext[AgentDeps], limit: int = 20) -> dict[str, Any]:
    """Find tracks that triggered calm or relaxed brain states.

    Queries historical EEG data to find tracks associated with
    high valence and low arousal (the relaxation quadrant).
    """
    relaxed = await eeg_service.find_tracks_by_mood(ctx.deps.db, "relaxed", limit=limit)
    calm = await eeg_service.find_tracks_by_mood(ctx.deps.db, "calm", limit=limit)

    all_tracks = relaxed + calm
    if not all_tracks:
        return {"tracks": [], "count": 0, "message": "No tracks with relaxation data found."}

    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for t in all_tracks:
        tid = str(t["track_id"])
        if tid not in seen:
            seen.add(tid)
            unique.append(t)

    return {
        "tracks": [
            {
                "title": t["title"],
                "artist": t["artist"],
                "dominant_state": t["dominant_state"],
                "avg_arousal": round(float(t["avg_arousal"]), 2),  # type: ignore[arg-type]
                "avg_valence": round(float(t["avg_valence"]), 2),  # type: ignore[arg-type]
            }
            for t in unique[:limit]
        ],
        "count": len(unique[:limit]),
        "total_available": len(unique),
    }


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
        return {
            "error": "confirmation_required",
            "message": "You must get explicit user confirmation before creating a playlist. "
            "Propose the playlist name and track selection, then call this tool "
            "with user_confirmed=True after they approve.",
        }

    tracks = await eeg_service.find_tracks_by_mood(ctx.deps.db, mood, limit=30)

    if not tracks:
        return {"error": f"No tracks found for mood '{mood}'. Try a different mood or seed more data."}

    playlist_name = name or f"CortexDJ: {mood.capitalize()} Mix"

    internal_ids = [str(t["track_id"]) for t in tracks]
    db_tracks = await Track.get_many(ctx.deps.db, internal_ids)
    spotify_track_ids = [t.spotify_track_id for t in db_tracks if t.spotify_track_id]

    playlist = Playlist(
        id=str(uuid.uuid4()),
        name=playlist_name,
        mood_criteria={"target_mood": mood, "track_count": len(tracks)},
        track_count=len(tracks),
    )
    ctx.deps.db.add(playlist)
    await ctx.deps.db.flush()

    spotify_result = await create_playlist(
        playlist_name,
        spotify_track_ids,
        description=f"Auto-curated by CortexDJ based on brain state: {mood}",
        client=ctx.deps.spotify_client,
    )

    if spotify_result.get("spotify_playlist_id"):
        playlist.spotify_playlist_id = spotify_result["spotify_playlist_id"]
        await ctx.deps.db.flush()

    return {
        "playlist_name": playlist_name,
        "mood": mood,
        "track_count": len(tracks),
        "tracks": [{"title": t["title"], "artist": t["artist"]} for t in tracks[:15]],
        "tracks_truncated": len(tracks) > 15,
        "spotify_url": spotify_result.get("spotify_url"),
        "spotify_playlist_id": spotify_result.get("spotify_playlist_id"),
        "note": spotify_result.get("note"),
    }


# ---------------------------------------------------------------------------
# Spotify user-authenticated tools (hidden when not connected)
# ---------------------------------------------------------------------------


async def get_listening_history(ctx: RunContext[AgentDeps], limit: int = 20) -> dict[str, Any]:
    """Get recent Spotify listening history.

    Requires Spotify to be connected. Returns recently played tracks
    which can be correlated with EEG session data.
    """
    if ctx.deps.spotify_client is None:
        return _requires_connection_error()

    try:
        results = await run_spotify(ctx.deps.spotify_client.current_user_recently_played, limit=limit)
    except spotipy.SpotifyException as e:
        if e.http_status == 401:
            return _token_expired_error()
        return _spotify_api_error("get listening history", e)
    except Exception as e:
        return _spotify_api_error("get listening history", e)

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

    Requires Spotify to be connected. Useful for finding existing playlists
    or correlating with EEG session data.
    """
    client = ctx.deps.spotify_client
    if client is None:
        return _requires_connection_error()

    try:
        items, total = await fetch_all_pages(
            client.current_user_playlists,
            limit=50,
            max_items=max_results,
        )

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
    except spotipy.SpotifyException as e:
        if e.http_status == 401:
            return _token_expired_error()
        return _spotify_api_error("get playlists", e)
    except Exception as e:
        return _spotify_api_error("get playlists", e)


async def get_my_saved_tracks(
    ctx: RunContext[AgentDeps],
    max_results: int | None = 50,
) -> dict[str, Any]:
    """Get tracks saved in the user's Spotify library.

    Requires Spotify to be connected. Useful for cross-referencing
    the user's library with EEG session data.
    """
    client = ctx.deps.spotify_client
    if client is None:
        return _requires_connection_error()

    try:
        items, total = await fetch_all_pages(
            client.current_user_saved_tracks,
            limit=50,
            max_items=max_results,
        )

        response: dict[str, Any] = {
            "saved_tracks": [
                {
                    "name": item["track"]["name"],
                    "artists": [a["name"] for a in item["track"]["artists"]],
                    "track_id": item["track"]["id"],
                    "added_at": item.get("added_at"),
                }
                for item in items
            ],
            "total": total,
            "fetched_count": len(items),
        }
        if max_results is not None and len(items) >= max_results and len(items) < total:
            response["capped"] = True
        return response
    except spotipy.SpotifyException as e:
        if e.http_status == 401:
            return _token_expired_error()
        return _spotify_api_error("get saved tracks", e)
    except Exception as e:
        return _spotify_api_error("get saved tracks", e)


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
        return {
            "error": "confirmation_required",
            "message": "You must get explicit user confirmation before adding tracks to a playlist. "
            "Tell the user which tracks you plan to add and to which playlist, then call "
            "with user_confirmed=True after they approve.",
        }

    client = ctx.deps.spotify_client
    if client is None:
        return _requires_connection_error()

    if not track_ids:
        return {"error": "No track IDs provided. Please specify at least one track to add."}

    if len(track_ids) > 100:
        return {
            "error": f"Too many tracks ({len(track_ids)}). Maximum 100 tracks per call. "
            "Split into multiple calls for larger additions."
        }

    try:
        track_uris = [f"spotify:track:{track_id}" for track_id in track_ids]
        await run_spotify(client.playlist_add_items, playlist_id, track_uris)

        return {
            "tracks_added": len(track_ids),
            "playlist_id": playlist_id,
        }
    except spotipy.SpotifyException as e:
        if e.http_status == 404:
            return {"error": f"Playlist not found: {playlist_id}"}
        if e.http_status == 403:
            return {
                "error": "permission_denied",
                "message": "You don't have permission to modify this playlist. "
                "You can only add tracks to playlists you own.",
            }
        if e.http_status == 401:
            return _token_expired_error()
        return _spotify_api_error("add tracks", e)
    except Exception as e:
        return _spotify_api_error("add tracks", e)


# ---------------------------------------------------------------------------
# Public Spotify tools (Client Credentials, no user auth needed)
# ---------------------------------------------------------------------------


async def search_tracks(
    query: str,
    max_results: int = 10,
) -> dict[str, Any]:
    """Search for tracks on Spotify by name, artist, or keywords.

    Args:
        query: Search query (e.g., "Bohemian Rhapsody", "Queen", "chill electronic")
        max_results: Total tracks to return (default 10, max 200)
    """
    client = get_spotify_client()
    if client is None:
        return _spotify_not_configured_error()

    max_results = max(1, min(200, max_results))

    try:
        all_tracks, total_available = await search_paginated(client, query, "track", max_results)

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
    except Exception as e:
        return _spotify_api_error("search tracks", e)


async def get_track_info(
    track_id: str,
) -> dict[str, Any]:
    """Get detailed information about a specific Spotify track.

    Args:
        track_id: The Spotify track ID
    """
    client = get_spotify_client()
    if client is None:
        return _spotify_not_configured_error()

    try:
        track = await run_spotify(client.track, track_id)

        return {
            "name": track["name"],
            "artists": [a["name"] for a in track["artists"]],
            "album": track["album"]["name"],
            "release_date": track["album"].get("release_date"),
            "duration_ms": track["duration_ms"],
            "popularity": track["popularity"],
            "explicit": track["explicit"],
            "spotify_url": track["external_urls"].get("spotify"),
        }
    except Exception as e:
        return _spotify_api_error("get track info", e)
