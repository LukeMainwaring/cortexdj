"""Agent type enum for conversation namespacing."""

from enum import StrEnum


class AgentType(StrEnum):
    """Valid agent types for namespacing message history."""

    CHAT = "chat"
