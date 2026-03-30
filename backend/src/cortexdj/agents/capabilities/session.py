"""Session analysis capability."""

from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from cortexdj.agents.deps import AgentDeps
from cortexdj.agents.tools.session_tools import analyze_session, list_sessions


@dataclass
class SessionCapability(AbstractCapability[AgentDeps]):
    """EEG session listing and analysis tools."""

    def get_toolset(self) -> FunctionToolset[AgentDeps]:
        ts: FunctionToolset[AgentDeps] = FunctionToolset()
        ts.tool(list_sessions)
        ts.tool(analyze_session)
        return ts
