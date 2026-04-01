from dataclasses import dataclass

from pydantic_ai import RunContext, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from cortexdj.agents.deps import AgentDeps
from cortexdj.agents.tools.playlist_tools import (
    build_mood_playlist,
    find_relaxing_tracks,
    get_listening_history,
)

_SPOTIFY_ONLY_TOOLS = frozenset({get_listening_history.__name__})


@dataclass
class PlaylistCapability(AbstractCapability[AgentDeps]):
    def get_toolset(self) -> FunctionToolset[AgentDeps]:
        ts: FunctionToolset[AgentDeps] = FunctionToolset()
        ts.tool(find_relaxing_tracks)
        ts.tool(build_mood_playlist)
        ts.tool(get_listening_history)
        return ts

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDeps],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Hide Spotify-only tools when no Spotify client is available."""
        if ctx.deps.spotify_client is None:
            return [td for td in tool_defs if td.name not in _SPOTIFY_ONLY_TOOLS]
        return tool_defs
