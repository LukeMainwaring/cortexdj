"""Spotify for the assistant: mood playlists, track search, and the user's own library.

Every tool here runs on the user's own Spotify login (``client.get_spotify_client``), so
without one ``prepare_tools`` hides them all and the model never offers a tool that can
only fail. That is decided from the database session on every model step. The
instructions are all static, so they sit in the cached prefix
(``docs/adr/0002-prompt-caching-in-the-shared-recipe.md``); they say what to tell the
user when the tools are missing.
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
    search_tracks,
)

_SPOTIFY_INSTRUCTIONS = """
## Spotify Tools

Offered only while the user's Spotify account is connected. If they are not offered and the
user asks for one, say Spotify is not connected and point them to Settings > Spotify.

- **build_mood_playlist** — Build a Spotify playlist from tracks matching a brain-derived mood (requires user confirmation)
- **search_tracks** — Search Spotify for tracks by name, artist, or keywords
- **get_listening_history** — Fetch recently played tracks from Spotify
- **get_my_playlists** — Browse the user's Spotify playlists
- **add_tracks_to_playlist** — Add tracks to an existing Spotify playlist (requires user confirmation)

- For playlist building, explain the brain-music connection

## Playlist Creation
- ALWAYS ask the user for confirmation before calling build_mood_playlist or add_tracks_to_playlist
- For build_mood_playlist: propose the playlist name and mood first, then call with user_confirmed=True after approval
- For add_tracks_to_playlist: tell the user which tracks and playlist, then call with user_confirmed=True after approval
- Prefer add_tracks_to_playlist for existing playlists rather than creating new ones
""".strip()

_TOOLS = (
    build_mood_playlist,
    search_tracks,
    get_listening_history,
    get_my_playlists,
    add_tracks_to_playlist,
)
_TOOL_NAMES = frozenset(tool.__name__ for tool in _TOOLS)


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
        }
    )

    def get_toolset(self) -> FunctionToolset[AgentDeps]:
        # The tools share one AgentDeps.db session, and an AsyncSession cannot run
        # overlapping operations; pydantic-ai would otherwise run a response's several
        # tool calls concurrently.
        ts: FunctionToolset[AgentDeps] = FunctionToolset(sequential=True)
        for tool in _TOOLS:
            ts.tool(tool)
        return ts

    def get_instructions(self) -> str:
        return _SPOTIFY_INSTRUCTIONS

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDeps],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Hide the Spotify tools unless the user's Spotify login is connected."""
        if await get_spotify_client(ctx.deps.db) is None:
            return [td for td in tool_defs if td.name not in _TOOL_NAMES]
        return tool_defs
