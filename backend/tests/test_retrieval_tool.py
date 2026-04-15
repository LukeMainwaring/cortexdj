"""Unit tests for agents.tools.retrieval_tools.retrieve_tracks_from_brain_state.

The tool is a thin JSON-serializer around `services.retrieval`, but a couple
of behaviors are worth pinning: the populated-index success shape, the
empty-index `note` field (the agent's escape hatch when the operator hasn't
run `seed-track-index`), and exception propagation per the project's
pydantic-ai convention (let hooks handle tool errors, don't swallow them).
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cortexdj.agents.tools.retrieval_tools import retrieve_tracks_from_brain_state
from cortexdj.services.retrieval import DeapFileMissingError, TrackHit


def _make_ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.deps.db = MagicMock()
    return ctx


class TestRetrieveTracksFromBrainState:
    def test_populated_index_returns_json_with_ranked_hits(self) -> None:
        hits = [
            TrackHit(
                spotify_id="spid1",
                title="Song A",
                artist="Artist A",
                itunes_preview_url="https://example.com/a.m4a",
                audio_cache_key="a" * 40,
                similarity=0.87,
            ),
            TrackHit(
                spotify_id="spid2",
                title="Song B",
                artist="Artist B",
                itunes_preview_url=None,
                audio_cache_key=None,
                similarity=0.42,
            ),
        ]
        with patch(
            "cortexdj.agents.tools.retrieval_tools.retrieval_service.retrieve_similar_tracks",
            new=AsyncMock(return_value=hits),
        ):
            payload_json = asyncio.run(retrieve_tracks_from_brain_state(_make_ctx(), "sess-1", k=5))

        payload = json.loads(payload_json)
        assert payload["session_id"] == "sess-1"
        assert payload["k"] == 5
        assert len(payload["tracks"]) == 2
        assert payload["tracks"][0]["spotify_id"] == "spid1"
        assert payload["tracks"][0]["similarity"] == 0.87
        assert payload["tracks"][1]["itunes_preview_url"] is None
        # `note` is only set on the empty-index path; never leak it on success.
        assert "note" not in payload

    def test_empty_index_returns_note_field(self) -> None:
        with patch(
            "cortexdj.agents.tools.retrieval_tools.retrieval_service.retrieve_similar_tracks",
            new=AsyncMock(return_value=[]),
        ):
            payload_json = asyncio.run(retrieve_tracks_from_brain_state(_make_ctx(), "sess-1", k=10))

        payload = json.loads(payload_json)
        assert payload["tracks"] == []
        assert payload["k"] == 10
        # The note steers the agent into telling the user to run the seed
        # command instead of hallucinating a recovery.
        assert "seed-track-index" in payload["note"]

    def test_lookup_error_propagates_to_hooks(self) -> None:
        # Per .claude/rules/backend/pydantic-ai.md: tools let exceptions
        # propagate so on_tool_execute_error can produce a structured
        # recovery payload. The tool must NOT catch LookupError.
        with patch(
            "cortexdj.agents.tools.retrieval_tools.retrieval_service.retrieve_similar_tracks",
            new=AsyncMock(side_effect=LookupError("session sess-bogus not found")),
        ):
            with pytest.raises(LookupError, match="sess-bogus"):
                asyncio.run(retrieve_tracks_from_brain_state(_make_ctx(), "sess-bogus", k=5))

    def test_deap_file_missing_returns_structured_error(self) -> None:
        # `DeapFileMissingError` is server misconfig — the hooks recovery
        # template strips the exception message, so we catch it specifically
        # and return an actionable JSON payload the agent can relay verbatim.
        with patch(
            "cortexdj.agents.tools.retrieval_tools.retrieval_service.retrieve_similar_tracks",
            new=AsyncMock(side_effect=DeapFileMissingError("DEAP file for P99 not found at /x/s99.dat")),
        ):
            payload_json = asyncio.run(retrieve_tracks_from_brain_state(_make_ctx(), "sess-1", k=5))

        payload = json.loads(payload_json)
        assert payload["error"] == "deap_data_missing"
        assert "s99.dat" in payload["message"]
        assert "DEAP_SETUP.md" in payload["message"]
        # No `tracks` key on the error path — the agent should not try to
        # render an empty list as if retrieval succeeded.
        assert "tracks" not in payload

    def test_passes_k_through_to_service(self) -> None:
        mock = AsyncMock(return_value=[])
        with patch("cortexdj.agents.tools.retrieval_tools.retrieval_service.retrieve_similar_tracks", new=mock):
            asyncio.run(retrieve_tracks_from_brain_state(_make_ctx(), "sess-1", k=25))
        assert mock.call_args.kwargs["k"] == 25

    def test_negative_k_passes_through_unclamped_at_tool_layer(self) -> None:
        # Pins the "clamping lives in the service, not the tool" contract —
        # otherwise a well-meaning future refactor might add `max(1, k)` at
        # the tool layer and silently mask bugs in the service-side clamp.
        mock = AsyncMock(return_value=[])
        with patch("cortexdj.agents.tools.retrieval_tools.retrieval_service.retrieve_similar_tracks", new=mock):
            asyncio.run(retrieve_tracks_from_brain_state(_make_ctx(), "sess-1", k=-1))
        assert mock.call_args.kwargs["k"] == -1
