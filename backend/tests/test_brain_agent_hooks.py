"""Unit tests for the brain_agent hooks module.

Exercises the recovery payload helper and the ``_recover_tool_error``
handler directly against real ``ToolCallPart`` / ``ToolDefinition``
instances. End-to-end validation (fake failing tool → agent responds
conversationally) lives in ``tests/evals/``.
"""

from __future__ import annotations

import asyncio

from pydantic_ai import ToolDefinition
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ToolCallPart

from cortexdj.agents.hooks import (
    _recover_tool_error,
    _recovery_payload,
    build_brain_agent_hooks,
)


class TestRecoveryPayload:
    def test_contains_tool_and_exception_type(self) -> None:
        payload = _recovery_payload("get_my_playlists", RuntimeError("boom"))
        assert payload["error"] == "tool_failed"
        assert payload["tool"] == "get_my_playlists"
        assert payload["exception_type"] == "RuntimeError"
        assert "get_my_playlists" in payload["message"]

    def test_payload_shape_is_stable_across_exception_types(self) -> None:
        payload_a = _recovery_payload("search_tracks", ValueError("bad"))
        payload_b = _recovery_payload("search_tracks", ConnectionError("net"))
        assert payload_a.keys() == payload_b.keys()
        assert payload_a["exception_type"] == "ValueError"
        assert payload_b["exception_type"] == "ConnectionError"


class TestRecoverToolErrorHandler:
    def test_returns_recovery_payload_for_unexpected_exception(self) -> None:
        call = ToolCallPart(tool_name="search_tracks", tool_call_id="call-1")
        tool_def = ToolDefinition(name="search_tracks")

        async def _invoke() -> dict[str, object]:
            return await _recover_tool_error(
                None,  # type: ignore[arg-type]  # handler doesn't touch ctx
                call=call,
                tool_def=tool_def,
                args={},
                error=RuntimeError("spotify 500"),
            )

        result = asyncio.run(_invoke())
        assert result["error"] == "tool_failed"
        assert result["tool"] == "search_tracks"
        assert result["exception_type"] == "RuntimeError"


class TestBuildBrainAgentHooks:
    def test_returns_hooks_instance(self) -> None:
        hooks = build_brain_agent_hooks()
        assert isinstance(hooks, Hooks)
