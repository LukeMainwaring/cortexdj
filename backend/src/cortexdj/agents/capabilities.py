"""Agent capability registry (composition root).

The single registration point for the agent's capabilities: add a capability
here (defined in its feature slice under ``features/<name>/capability.py``) to
enable it, or remove the entry to disable it — no edits to ``assistant.py``
needed. As a composition-root module this is permitted to import feature
capabilities, which the dependency-direction rule otherwise forbids the agent
runtime from doing.
"""

from pydantic_ai.capabilities import AgentCapability

from cortexdj.agents.deps import AgentDeps
from cortexdj.features.retrieval.capability import RetrievalCapability
from cortexdj.features.sessions.capability import SessionsCapability
from cortexdj.features.spotify.capability import SpotifyCapability

ENABLED_CAPABILITIES: list[AgentCapability[AgentDeps]] = [
    SessionsCapability(),
    SpotifyCapability(),
    RetrievalCapability(),
]

__all__ = ["ENABLED_CAPABILITIES"]
