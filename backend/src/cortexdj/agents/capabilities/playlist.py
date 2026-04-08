from dataclasses import dataclass

from pydantic_ai import RunContext, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from cortexdj.agents.deps import AgentDeps
from cortexdj.agents.tools.playlist_tools import (
    add_tracks_to_playlist,
    build_mood_playlist,
    find_relaxing_tracks,
    get_listening_history,
    get_my_playlists,
    get_my_saved_tracks,
    get_track_info,
    search_tracks,
)

_SPOTIFY_ONLY_TOOLS = frozenset(
    {
        get_listening_history.__name__,
        get_my_playlists.__name__,
        get_my_saved_tracks.__name__,
        add_tracks_to_playlist.__name__,
    }
)


@dataclass
class PlaylistCapability(AbstractCapability[AgentDeps]):
    def get_toolset(self) -> FunctionToolset[AgentDeps]:
        ts: FunctionToolset[AgentDeps] = FunctionToolset()
        # EEG-based (always available)
        ts.tool(find_relaxing_tracks)
        ts.tool(build_mood_playlist)
        # Public Spotify (Client Credentials, always available)
        ts.tool_plain(search_tracks)
        ts.tool_plain(get_track_info)
        # User-authenticated Spotify (hidden when not connected)
        ts.tool(get_listening_history)
        ts.tool(get_my_playlists)
        ts.tool(get_my_saved_tracks)
        ts.tool(add_tracks_to_playlist)
        return ts

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDeps],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Hide Spotify user-authenticated tools when no Spotify client is available."""
        if ctx.deps.spotify_client is None:
            return [td for td in tool_defs if td.name not in _SPOTIFY_ONLY_TOOLS]
        return tool_defs
