"""Agent tools for Spotify playlist curation based on brain data."""

from __future__ import annotations

import uuid

from pydantic_ai import RunContext

from cortexdj.agents.deps import AgentDeps
from cortexdj.models.playlist import Playlist
from cortexdj.models.track import Track
from cortexdj.services import eeg_processing as eeg_service
from cortexdj.services import spotify as spotify_service


async def find_relaxing_tracks(ctx: RunContext[AgentDeps], limit: int = 20) -> str:
    """Find tracks that triggered calm or relaxed brain states.

    Queries historical EEG data to find tracks associated with
    high valence and low arousal (the relaxation quadrant).
    """
    relaxed = await eeg_service.find_tracks_by_mood(ctx.deps.db, "relaxed", limit=limit)
    calm = await eeg_service.find_tracks_by_mood(ctx.deps.db, "calm", limit=limit)

    all_tracks = relaxed + calm
    if not all_tracks:
        return "No tracks with relaxation data found. Make sure sessions are seeded with track associations."

    # Deduplicate by track_id
    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for t in all_tracks:
        if t["track_id"] not in seen:
            seen.add(str(t["track_id"]))
            unique.append(t)

    lines = [f"**Found {len(unique)} tracks associated with relaxed/calm brain states:**\n"]
    for t in unique[:limit]:
        lines.append(
            f"- **{t['title']}** by {t['artist']} | "
            f"State: {t['dominant_state']} | "
            f"Arousal: {t['avg_arousal']:.2f} | Valence: {t['avg_valence']:.2f}"
        )

    return "\n".join(lines)


async def build_mood_playlist(
    ctx: RunContext[AgentDeps],
    mood: str,
    name: str | None = None,
) -> str:
    """Build a Spotify playlist from tracks matching a brain-derived mood.

    Args:
        mood: Target mood (relaxed, calm, excited, stressed)
        name: Optional playlist name (auto-generated if not provided)
    """
    tracks = await eeg_service.find_tracks_by_mood(ctx.deps.db, mood, limit=30)

    if not tracks:
        return f"No tracks found for mood '{mood}'. Try a different mood or seed more data."

    playlist_name = name or f"CortexDJ: {mood.capitalize()} Mix"

    # Bulk look up Spotify track IDs for real playlist creation
    internal_ids = [str(t["track_id"]) for t in tracks]
    db_tracks = await Track.get_many(ctx.deps.db, internal_ids)
    spotify_track_ids = [t.spotify_track_id for t in db_tracks if t.spotify_track_id]

    # Record playlist in database
    playlist = Playlist(
        id=str(uuid.uuid4()),
        name=playlist_name,
        mood_criteria={"target_mood": mood, "track_count": len(tracks)},
        track_count=len(tracks),
    )
    ctx.deps.db.add(playlist)
    await ctx.deps.db.flush()

    # Create on Spotify if we have valid Spotify track IDs
    spotify_result = await spotify_service.create_playlist(
        playlist_name,
        spotify_track_ids,
        description=f"Auto-curated by CortexDJ based on brain state: {mood}",
        client=ctx.deps.spotify_client,
    )

    # Update playlist record with Spotify ID if created
    if spotify_result.get("spotify_playlist_id"):
        playlist.spotify_playlist_id = spotify_result["spotify_playlist_id"]
        await ctx.deps.db.flush()

    lines = [
        f"**Playlist Created: {playlist_name}**\n",
        f"Mood: {mood.capitalize()} | {len(tracks)} tracks\n",
    ]
    for i, t in enumerate(tracks[:15], 1):
        lines.append(f"{i}. **{t['title']}** by {t['artist']}")
    if len(tracks) > 15:
        lines.append(f"... and {len(tracks) - 15} more tracks")

    if spotify_result.get("spotify_url"):
        lines.append(f"\n[Open in Spotify]({spotify_result['spotify_url']})")
    elif spotify_result.get("note"):
        lines.append(f"\n*Note: {spotify_result['note']}*")

    return "\n".join(lines)


async def get_listening_history(ctx: RunContext[AgentDeps], limit: int = 20) -> str:
    """Get recent Spotify listening history.

    Requires Spotify to be connected. Returns recently played tracks
    which can be correlated with EEG session data.
    """
    if ctx.deps.spotify_client is None:
        return "Spotify is not connected. Connect Spotify in Settings to view listening history."

    try:
        results = await spotify_service.run_spotify(ctx.deps.spotify_client.current_user_recently_played, limit=limit)
    except Exception:
        return "Failed to fetch listening history. Try reconnecting Spotify in Settings."

    items = results.get("items", [])
    if not items:
        return "No recent listening history found."

    lines = [f"**Recently Played ({len(items)} tracks):**\n"]
    for item in items:
        track = item["track"]
        played_at = item.get("played_at", "")
        artists = ", ".join(a["name"] for a in track["artists"])
        lines.append(f"- **{track['name']}** by {artists} | {played_at}")

    return "\n".join(lines)
