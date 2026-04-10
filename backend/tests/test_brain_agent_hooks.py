"""Unit tests for the brain_agent hooks module.

Full end-to-end recovery testing (fake tool that raises → agent responds
conversationally) lives in ``tests/evals/`` because it requires standing up
a TestModel-backed agent. These tests cover the helpers and registration
wiring only.
"""

from __future__ import annotations

import asyncio

from pydantic_ai.capabilities import Hooks

from cortexdj.agents.hooks import _recovery_payload, build_brain_agent_hooks


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


class TestBuildBrainAgentHooks:
    def test_returns_hooks_instance(self) -> None:
        hooks = build_brain_agent_hooks()
        assert isinstance(hooks, Hooks)

    def test_registers_tool_execute_error_handler(self) -> None:
        hooks = build_brain_agent_hooks()
        # Internal registry key follows AbstractCapability method names; see
        # pydantic_ai/capabilities/hooks.py Hooks.__init__ for the mapping.
        entries = hooks._get("on_tool_execute_error")
        assert len(entries) == 1

    def test_does_not_register_unused_hooks(self) -> None:
        hooks = build_brain_agent_hooks()
        for key in (
            "before_run",
            "after_run",
            "wrap_run",
            "on_run_error",
            "before_model_request",
            "after_model_request",
        ):
            assert hooks._get(key) == []

    def test_registered_handler_returns_recovery_payload(self) -> None:
        hooks = build_brain_agent_hooks()
        entry = hooks._get("on_tool_execute_error")[0]

        class _FakeToolDef:
            name = "search_tracks"

        class _FakeCall:
            tool_name = "search_tracks"
            tool_call_id = "test-id"

        async def _invoke() -> object:
            return await entry.func(
                None,
                call=_FakeCall(),
                tool_def=_FakeToolDef(),
                args={},
                error=RuntimeError("spotify 500"),
            )

        result = asyncio.run(_invoke())
        assert isinstance(result, dict)
        assert result["error"] == "tool_failed"
        assert result["tool"] == "search_tracks"
        assert result["exception_type"] == "RuntimeError"
