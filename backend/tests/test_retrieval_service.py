"""Unit tests for services.retrieval.

The encoder loading, DEAP `.dat` parsing, and real pgvector cosine search
paths are exercised end-to-end via the MPS smoke run and the `seed-track-index`
integration test. These tests pin the pure-function helpers and the
orchestration logic's handling of edge cases (empty index, k clamping,
lookup failures, cosine→similarity conversion).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from cortexdj.services.retrieval import (
    TrackHit,
    _participant_dat_path,
    retrieve_similar_tracks,
    serialize_hits,
)


class TestParticipantDatPath:
    def test_parses_standard_format(self) -> None:
        path = _participant_dat_path("P01")
        assert path.name == "s01.dat"

    def test_parses_two_digit(self) -> None:
        path = _participant_dat_path("P32")
        assert path.name == "s32.dat"

    def test_rejects_missing_prefix(self) -> None:
        with pytest.raises(ValueError, match="unexpected participant_id format"):
            _participant_dat_path("01")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="unexpected participant_id format"):
            _participant_dat_path("")


class TestSerializeHits:
    def test_empty_list(self) -> None:
        assert serialize_hits([]) == []

    def test_rounds_similarity_to_4_decimals(self) -> None:
        hit = TrackHit(
            spotify_id="abc",
            title="Song",
            artist="Artist",
            itunes_preview_url="https://example.com/a.m4a",
            audio_cache_key="0" * 40,
            similarity=0.123456789,
        )
        serialized = serialize_hits([hit])
        assert serialized[0]["similarity"] == 0.1235
        assert serialized[0]["spotify_id"] == "abc"
        assert serialized[0]["title"] == "Song"
        assert serialized[0]["itunes_preview_url"] == "https://example.com/a.m4a"
        assert serialized[0]["audio_cache_key"] == "0" * 40

    def test_handles_null_preview_url(self) -> None:
        hit = TrackHit(
            spotify_id="x",
            title="t",
            artist="a",
            itunes_preview_url=None,
            audio_cache_key=None,
            similarity=0.5,
        )
        serialized = serialize_hits([hit])
        assert serialized[0]["itunes_preview_url"] is None
        assert serialized[0]["audio_cache_key"] is None


class TestRetrieveSimilarTracks:
    def test_missing_session_raises_lookup_error(self) -> None:
        db = MagicMock()
        with patch("cortexdj.services.retrieval.Session.get", new=AsyncMock(return_value=None)):
            with pytest.raises(LookupError, match="sess-bogus"):
                asyncio.run(retrieve_similar_tracks(db, "sess-bogus", k=10))

    def test_empty_index_returns_empty_list(self) -> None:
        db = MagicMock()
        with (
            patch("cortexdj.services.retrieval.Session.get", new=AsyncMock(return_value=MagicMock())),
            patch("cortexdj.services.retrieval.TrackAudioEmbedding.count", new=AsyncMock(return_value=0)),
            patch("cortexdj.services.retrieval.encode_session_to_clap_space") as mock_encode,
        ):
            hits = asyncio.run(retrieve_similar_tracks(db, "sess-1", k=10))
        assert hits == []
        # Importantly, we did NOT call the encoder if the index is empty —
        # encoding is the expensive part and there's nothing to query against.
        mock_encode.assert_not_called()

    def test_k_is_clamped_to_valid_range(self) -> None:
        db = MagicMock()
        mock_row = MagicMock(
            spotify_id="abc",
            title="Song",
            artist="Artist",
            itunes_preview_url=None,
        )
        with (
            patch("cortexdj.services.retrieval.Session.get", new=AsyncMock(return_value=MagicMock())),
            patch("cortexdj.services.retrieval.TrackAudioEmbedding.count", new=AsyncMock(return_value=1)),
            patch(
                "cortexdj.services.retrieval.encode_session_to_clap_space",
                new=AsyncMock(return_value=np.zeros(512, dtype=np.float32)),
            ),
            patch(
                "cortexdj.services.retrieval.TrackAudioEmbedding.get_top_k_similar",
                new=AsyncMock(return_value=[(mock_row, 0.2)]),
            ) as mock_topk,
        ):
            asyncio.run(retrieve_similar_tracks(db, "sess-1", k=500))
        # k=500 should be clamped to 100.
        assert mock_topk.call_args.kwargs["k"] == 100

    def test_k_floor_is_one(self) -> None:
        db = MagicMock()
        with (
            patch("cortexdj.services.retrieval.Session.get", new=AsyncMock(return_value=MagicMock())),
            patch("cortexdj.services.retrieval.TrackAudioEmbedding.count", new=AsyncMock(return_value=1)),
            patch(
                "cortexdj.services.retrieval.encode_session_to_clap_space",
                new=AsyncMock(return_value=np.zeros(512, dtype=np.float32)),
            ),
            patch(
                "cortexdj.services.retrieval.TrackAudioEmbedding.get_top_k_similar",
                new=AsyncMock(return_value=[]),
            ) as mock_topk,
        ):
            asyncio.run(retrieve_similar_tracks(db, "sess-1", k=0))
        assert mock_topk.call_args.kwargs["k"] == 1

    def test_cosine_distance_converts_to_similarity(self) -> None:
        # pgvector returns cosine distance in [0, 2] (0 = identical, 1 = orthogonal,
        # 2 = antipodal). We convert to similarity = 1 - distance so callers get
        # the standard [-1, 1] cosine-similarity convention where higher is closer.
        db = MagicMock()
        rows = [
            (MagicMock(spotify_id="a", title="Ta", artist="Aa", itunes_preview_url=None), 0.0),
            (MagicMock(spotify_id="b", title="Tb", artist="Ab", itunes_preview_url=None), 0.5),
            (MagicMock(spotify_id="c", title="Tc", artist="Ac", itunes_preview_url=None), 1.0),
        ]
        with (
            patch("cortexdj.services.retrieval.Session.get", new=AsyncMock(return_value=MagicMock())),
            patch("cortexdj.services.retrieval.TrackAudioEmbedding.count", new=AsyncMock(return_value=3)),
            patch(
                "cortexdj.services.retrieval.encode_session_to_clap_space",
                new=AsyncMock(return_value=np.zeros(512, dtype=np.float32)),
            ),
            patch(
                "cortexdj.services.retrieval.TrackAudioEmbedding.get_top_k_similar",
                new=AsyncMock(return_value=rows),
            ),
        ):
            hits = asyncio.run(retrieve_similar_tracks(db, "sess-1", k=3))
        similarities = [h.similarity for h in hits]
        assert similarities == [1.0, 0.5, 0.0]
