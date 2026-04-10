"""Deterministic tests for capability ``prepare_tools`` filtering.

These tests use pydantic-ai's ``TestModel`` so they don't call the real
OpenAI API. They verify that when the agent is invoked, the tool list it
receives from the model-request-preparation path correctly reflects the
current ``AgentDeps`` state (Spotify connected/disconnected, EEG model
loaded/missing).

This covers the ``prepare_tools`` overrides in
``PlaylistCapability`` and ``ClassificationCapability`` end-to-end through
the agent — the filter functions themselves are exercised by the real
model-routing harness, but the wiring they depend on is validated here.
"""

from __future__ import annotations

import asyncio

from pydantic_ai.models.test import TestModel

from cortexdj.agents.brain_agent import brain_agent
from tests.evals.conftest import make_fake_deps


def _offered_tool_names(model: TestModel) -> set[str]:
    params = model.last_model_request_parameters
    assert params is not None, "TestModel has no recorded request parameters"
    return {t.name for t in params.function_tools}


def _run_agent_with_test_model(
    *,
    spotify_client: object | None,
    eeg_model: object | None,
) -> TestModel:
    test_model = TestModel(call_tools=[], custom_output_text="ok")
    deps = make_fake_deps(spotify_client=spotify_client, eeg_model=eeg_model)

    async def _run() -> None:
        with brain_agent.override(model=test_model, deps=deps):
            await brain_agent.run("hello")

    asyncio.run(_run())
    return test_model


class TestPlaylistCapabilityPrepareTools:
    def test_hides_user_spotify_tools_when_disconnected(self) -> None:
        model = _run_agent_with_test_model(spotify_client=None, eeg_model=None)
        offered = _offered_tool_names(model)

        for hidden in (
            "get_my_playlists",
            "get_my_saved_tracks",
            "get_listening_history",
            "add_tracks_to_playlist",
        ):
            assert hidden not in offered, f"{hidden} should be hidden when spotify_client is None"

    def test_public_spotify_tools_always_available(self) -> None:
        model = _run_agent_with_test_model(spotify_client=None, eeg_model=None)
        offered = _offered_tool_names(model)

        assert "search_tracks" in offered
        assert "get_track_info" in offered

    def test_eeg_tools_always_available_regardless_of_spotify(self) -> None:
        model = _run_agent_with_test_model(spotify_client=None, eeg_model=None)
        offered = _offered_tool_names(model)

        assert "find_relaxing_tracks" in offered
        assert "build_mood_playlist" in offered

    def test_shows_user_spotify_tools_when_connected(self) -> None:
        fake_spotify = object()  # PlaylistCapability only checks truthiness
        model = _run_agent_with_test_model(spotify_client=fake_spotify, eeg_model=None)
        offered = _offered_tool_names(model)

        assert "get_my_playlists" in offered
        assert "add_tracks_to_playlist" in offered


class TestClassificationCapabilityPrepareTools:
    def test_hides_model_tools_when_eeg_model_missing(self) -> None:
        model = _run_agent_with_test_model(spotify_client=None, eeg_model=None)
        offered = _offered_tool_names(model)

        assert "get_model_info" not in offered

    def test_set_brain_context_always_available(self) -> None:
        model = _run_agent_with_test_model(spotify_client=None, eeg_model=None)
        offered = _offered_tool_names(model)

        assert "set_brain_context" in offered

    def test_shows_model_tools_when_eeg_model_loaded(self) -> None:
        fake_eeg_model = object()  # ClassificationCapability only checks truthiness
        model = _run_agent_with_test_model(spotify_client=None, eeg_model=fake_eeg_model)
        offered = _offered_tool_names(model)

        assert "get_model_info" in offered


class TestAlwaysAvailableTools:
    def test_session_and_insight_tools_always_present(self) -> None:
        model = _run_agent_with_test_model(spotify_client=None, eeg_model=None)
        offered = _offered_tool_names(model)

        for always in (
            "list_sessions",
            "analyze_session",
            "explain_brain_state",
            "compare_sessions",
        ):
            assert always in offered
