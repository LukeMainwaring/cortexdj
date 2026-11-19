"""The shared recipe every named agent is assembled with, and the assistant built on it.

``build_agent`` is the one place the runtime-wide pieces are wired: the tool error
hooks, the history processor, prompt caching, optional thinking, and the tool
timeout. A new agent (see ``agents/registry.py``) adds its capabilities and output
type and inherits the rest. This module is feature-agnostic: capabilities own
their domain-specific tools and instructions, so don't hardcode feature tool
names or guidance here.

Callers obtain agents by name through ``agents/registry.py`` (``get_agent``),
which builds once and caches. Do not call a builder directly outside the registry.
"""

import logging
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AgentCapability, Caching, ProcessHistory, Thinking
from pydantic_ai.output import OutputSpec

from cortexdj.agents.capabilities import ENABLED_CAPABILITIES
from cortexdj.agents.deps import AgentDeps
from cortexdj.agents.history_processor import build_tool_result_summarizer
from cortexdj.agents.hooks import build_agent_hooks
from cortexdj.core.config import get_settings

logger = logging.getLogger(__name__)

ASSISTANT_AGENT_NAME = "assistant"

# Static text: pydantic-ai places the provider's cache breakpoint after the last
# static instruction, so rules belong in a literal like this, and only what varies
# per run goes in a callable (docs/adr/0002-prompt-caching-in-the-shared-recipe.md).
SYSTEM_PROMPT = """You are CortexDJ, a friendly and helpful AI assistant.

Be concise and conversational. Use the tools available to you when they help
with the user's request; otherwise, just answer helpfully.
""".strip()

# A hung tool (e.g. a stuck DB query) would otherwise hang the response stream;
# on timeout the model gets a retry prompt instead.
TOOL_TIMEOUT_SECONDS = 30


def build_agent(
    *,
    name: str,
    model: str,
    instructions: str | None,
    capabilities: Sequence[AgentCapability[AgentDeps]],
    output_type: OutputSpec[Any] = str,
    cache_messages: bool = True,
) -> Agent[AgentDeps, Any]:
    """Assemble an agent from its capabilities on top of the shared runtime wiring.

    ``model`` is the caller's: each builder resolves its own setting with
    ``Settings.resolve_model``, so per-agent models never touch this recipe.
    ``cache_messages=False`` caches only the tools and static instructions, for an
    agent that makes one request per run (ADR 0002).

    Called lazily by the registry so the app boots without a provider API key
    configured: the key is only needed when a chat request actually runs the
    agent, not at import.
    """
    # Each capability declares which of its tools return large results worth
    # compacting in history; the runtime stays free of feature tool names.
    summarizable_tools: set[str] = set()
    for capability in capabilities:
        summarizable_tools.update(getattr(capability, "summarizable_tools", frozenset()))

    all_capabilities: list[AgentCapability[AgentDeps]] = [
        *capabilities,
        build_agent_hooks(),
        ProcessHistory(build_tool_result_summarizer(frozenset(summarizable_tools))),
        Caching(messages=cache_messages),  # ADR 0002
    ]
    # Only when configured (default None = provider default). Unsupported levels
    # map to the closest one the model offers, so any value is safe on any model.
    reasoning_effort = get_settings().AGENT_REASONING_EFFORT
    if reasoning_effort is not None:
        all_capabilities.append(Thinking(effort=reasoning_effort))
    return Agent(
        model=model,
        name=name,  # labels the run span in Logfire
        deps_type=AgentDeps,
        output_type=output_type,
        instructions=instructions,
        capabilities=all_capabilities,
        tool_timeout=TOOL_TIMEOUT_SECONDS,
    )


# The model is provider-agnostic: pydantic-ai infers the provider from the
# `AGENT_MODEL` prefix ("anthropic:..." or "openai:...").
def build_assistant() -> Agent[AgentDeps, Any]:
    """The chat assistant: every enabled capability on the shared recipe."""
    return build_agent(
        name=ASSISTANT_AGENT_NAME,
        model=get_settings().AGENT_MODEL,
        instructions=SYSTEM_PROMPT,
        capabilities=ENABLED_CAPABILITIES,
    )
