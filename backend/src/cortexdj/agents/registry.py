"""Named-agent registry: the one lookup the app uses to obtain an agent.

Agents are registered by name with a builder; ``get_agent`` builds each one
lazily on first use and caches it, because building constructs a provider
client and must not happen at import. Callers (the chat stream, title
generation) never import a builder directly: they ask for a name, so adding an
agent is a one-line change here and needs no edit at the call sites. Which name
a request should use is the caller's decision; keep selection logic out of this
module.

This is a composition root (like ``api.py`` and ``agents/capabilities.py``): it
may import a feature's agent builder, which the dependency-direction rule
otherwise forbids the runtime from doing. Nothing else in ``agents/`` may.

Tests fake an agent through ``get_agent(NAME).override(model=...)``, never by
replacing an entry in ``AGENT_BUILDERS``: the built agent is cached for the
process, so a swapped builder would outlive its test
(docs/adr/0001-test-tiers-and-fakes.md).
"""

from collections.abc import Callable
from typing import Any

from pydantic_ai import Agent

from cortexdj.agents.assistant import ASSISTANT_AGENT_NAME, build_assistant
from cortexdj.agents.title_agent import THREAD_TITLE_AGENT_NAME, build_title_agent

ASSISTANT = ASSISTANT_AGENT_NAME
THREAD_TITLE = THREAD_TITLE_AGENT_NAME

# Deps differ per agent (the assistant takes AgentDeps, the title agent none),
# so the registry is typed loosely; callers pass the deps the agent they asked
# for expects.
AgentBuilder = Callable[[], Agent[Any, Any]]

_built: dict[str, Agent[Any, Any]] = {}


class UnknownAgent(LookupError):
    def __init__(self, name: str):
        super().__init__(f"No agent registered under {name!r}; known: {sorted(AGENT_BUILDERS)}")


def get_agent(name: str) -> Agent[Any, Any]:
    if name not in AGENT_BUILDERS:
        raise UnknownAgent(name)
    if name not in _built:
        _built[name] = AGENT_BUILDERS[name]()
    return _built[name]


# Defined after get_agent so an entry may reach another agent through
# `partial(get_agent, NAME)`, the way a specialist is handed to the capability
# that delegates to it.
AGENT_BUILDERS: dict[str, AgentBuilder] = {
    ASSISTANT: build_assistant,
    THREAD_TITLE: build_title_agent,
}
