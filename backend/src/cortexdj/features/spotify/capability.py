"""Spotify for the assistant: mood playlists, track search, and the user's own library.

Which Spotify client a call uses is ``client.get_spotify_client``'s decision, made from
the database session on every model step: the four library tools need the user's own
login and are hidden without it, so the model never offers a tool that can only fail.
The instructions are all static, so they sit in the cached prefix
(``docs/adr/0002-prompt-caching-in-the-shared-recipe.md``).
"""

from dataclasses import dataclass

from pydantic_ai import RunContext, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from cortexdj.agents.deps import AgentDeps
from cortexdj.features.spotify.client import get_spotify_client
from cortexdj.features.spotify.tools import (
    add_tracks_to_playlist,
    build_mood_playlist,
    get_listening_history,
    get_my_playlists,
    get_my_saved_tracks,
    get_track_info,
    search_tracks,
)

_SPOTIFY_INSTRUCTIONS = """
## Playlist & Track Tools

- **build_mood_playlist** — Build a playlist from tracks matching a brain-derived mood (requires user confirmation)
- **search_tracks** — Search Spotify for tracks by name, artist, or keywords
- **get_track_info** — Get detailed metadata for a specific Spotify track

## Spotify Library (requires Spotify connection)

- **get_listening_history** — Fetch recently played tracks from Spotify
- **get_my_playlists** — Browse the user's Spotify playlists
- **get_my_saved_tracks** — Access the user's saved Spotify library
- **add_tracks_to_playlist** — Add tracks to an existing Spotify playlist

- For playlist building, explain the brain-music connection

## Playlist Creation
- ALWAYS ask the user for confirmation before calling build_mood_playlist or add_tracks_to_playlist
- For build_mood_playlist: propose the playlist name and mood first, then call with user_confirmed=True after approval
- For add_tracks_to_playlist: tell the user which tracks and playlist, then call with user_confirmed=True after approval
- Prefer add_tracks_to_playlist for existing playlists rather than creating new ones
""".strip()

_SPOTIFY_ONLY_TOOLS = frozenset(
    {
        get_listening_history.__name__,
        get_my_playlists.__name__,
        get_my_saved_tracks.__name__,
        add_tracks_to_playlist.__name__,
    }
)


@dataclass
class SpotifyCapability(AbstractCapability[AgentDeps]):
    # Tools whose large list results the history processor may compact in prior
    # turns (see agents/history_processor.py).
    summarizable_tools: frozenset[str] = frozenset(
        {
            build_mood_playlist.__name__,
            get_listening_history.__name__,
            search_tracks.__name__,
            get_my_playlists.__name__,
            get_my_saved_tracks.__name__,
        }
    )

    def get_toolset(self) -> FunctionToolset[AgentDeps]:
        # The tools share one AgentDeps.db session, and an AsyncSession cannot run
        # overlapping operations; pydantic-ai would otherwise run a response's several
        # tool calls concurrently.
        ts: FunctionToolset[AgentDeps] = FunctionToolset(sequential=True)
        ts.tool(build_mood_playlist)
        ts.tool(search_tracks)
        ts.tool(get_track_info)
        # Hidden by prepare_tools when the user has not connected Spotify
        ts.tool(get_listening_history)
        ts.tool(get_my_playlists)
        ts.tool(get_my_saved_tracks)
        ts.tool(add_tracks_to_playlist)
        return ts

    def get_instructions(self) -> str:
        return _SPOTIFY_INSTRUCTIONS

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDeps],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Hide the Spotify library tools unless the user's own Spotify login is connected."""
        if not (await get_spotify_client(ctx.deps.db)).is_user:
            return [td for td in tool_defs if td.name not in _SPOTIFY_ONLY_TOOLS]
        return tool_defs
