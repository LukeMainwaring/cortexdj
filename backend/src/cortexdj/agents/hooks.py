"""Hooks for the CortexDJ brain agent.

Tool bodies already handle anticipated failures (Spotify not configured,
token expired, etc.) by returning structured ``{"error": ...}`` dicts —
those are ordinary domain results, not failures.

This module is the safety net for *unanticipated* exceptions — anything
that bubbles out of a tool body would otherwise abort the run and crash the
Vercel AI SDK stream mid-response. ``on_tool_execute_error`` logs the
traceback and raises ``ToolFailed``, which Pydantic AI records as a native
failed tool outcome (``ToolReturnPart.outcome == "failed"``) without spending
the tool's retry budget, so the model adapts conversationally instead of
re-issuing the same doomed call.

This is layer 2 of the two-layer tool-error convention; see
docs/adr/0003-two-layer-tool-error-convention.md for why tools propagate
rather than wrap, why the failure message never carries the exception's own
text, and for the one sanctioned inline catch.
"""

import logging
from typing import Any, NoReturn

from pydantic_ai import ToolDefinition
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import RunContext

from cortexdj.agents.deps import AgentDeps

logger = logging.getLogger(__name__)


async def _recover_tool_error(
    ctx: RunContext[AgentDeps],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: Any,
    error: Exception,
) -> NoReturn:
    # The traceback goes to the logs, never to the model: an exception message can
    # carry connection strings, tokens, or upstream response bodies. The model gets
    # the tool name and the exception class, which is enough to explain and move on.
    logger.exception(f"Unhandled exception in tool {tool_def.name}: {error!r}")
    raise ToolFailed(
        f"The {tool_def.name} tool failed unexpectedly ({type(error).__name__}). "
        "Apologize to the user, explain briefly what you were trying to do, "
        "and suggest they retry or rephrase."
    )


def build_brain_agent_hooks() -> Hooks[AgentDeps]:
    return Hooks[AgentDeps](tool_execute_error=_recover_tool_error)
