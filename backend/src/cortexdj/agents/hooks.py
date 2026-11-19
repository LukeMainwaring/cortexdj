"""Error hooks every agent gets through ``build_agent``.

Tool error convention: a tool raises ``ModelRetry`` when the model should
correct its arguments and try again, or ``ToolFailed`` when the call is done
and has failed and the model should see that and adapt. Tools never return
``{"error": ...}`` dicts.

This module is the safety net for everything else. Any other exception that
escapes a tool body would abort the run and crash the Vercel AI SDK stream
mid-response. ``on_tool_execute_error`` logs the traceback and converts it to
``ToolFailed``, so the failure is recorded in history (``outcome='failed'``),
traced as an error, and explained to the user conversationally.

``on_output_process_error`` is the same net for the output side: an exception
from an output function or an output validator is logged and becomes
``ModelRetry``, so the model answers again instead of the stream dying. Output
has no failed-result equivalent, so the retry spends the output retry budget.
It does not cover ``after_output_process`` hooks, which run outside the
recovery step, nor schema validation, which already retries on its own.
"""

import logging
from typing import Any

from pydantic_ai import ToolDefinition
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.output import OutputContext
from pydantic_ai.tools import RunContext

from cortexdj.agents.deps import AgentDeps

logger = logging.getLogger(__name__)


async def on_tool_execute_error(
    ctx: RunContext[AgentDeps],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, Any],
    error: Exception,
) -> Any:
    logger.exception(f"Unhandled exception in tool {tool_def.name}: {error!r}")
    raise ToolFailed(
        f"The {tool_def.name} tool failed unexpectedly ({type(error).__name__}). "
        "Tell the user briefly what you were trying to do and suggest they retry or rephrase."
    )


async def on_output_process_error(
    ctx: RunContext[AgentDeps],
    *,
    output_context: OutputContext,
    output: Any,
    error: Exception,
) -> Any:
    logger.exception(f"Unhandled exception processing output: {error!r}")
    raise ModelRetry(
        f"Processing your final answer failed unexpectedly ({type(error).__name__}). "
        "Answer again; if it fails again, tell the user briefly what you were trying to do."
    )


def build_agent_hooks() -> Hooks[AgentDeps]:
    return Hooks[AgentDeps](tool_execute_error=on_tool_execute_error, output_process_error=on_output_process_error)
