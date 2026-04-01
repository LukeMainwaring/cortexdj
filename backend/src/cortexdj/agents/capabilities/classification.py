from dataclasses import dataclass

from pydantic_ai import RunContext, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from cortexdj.agents.deps import AgentDeps
from cortexdj.agents.tools.classification_tools import get_model_info, set_brain_context

_MODEL_TOOLS = frozenset({get_model_info.__name__})


@dataclass
class ClassificationCapability(AbstractCapability[AgentDeps]):
    def get_toolset(self) -> FunctionToolset[AgentDeps]:
        ts: FunctionToolset[AgentDeps] = FunctionToolset()
        ts.tool(get_model_info)
        ts.tool(set_brain_context)
        return ts

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDeps],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Hide model tools when no EEG model is loaded."""
        if ctx.deps.eeg_model is None:
            return [td for td in tool_defs if td.name not in _MODEL_TOOLS]
        return tool_defs
