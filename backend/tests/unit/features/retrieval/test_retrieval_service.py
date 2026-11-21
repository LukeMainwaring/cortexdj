"""Pure helpers of the retrieval service: the DEAP file a participant id names, and the tool payload's hit shape.

Retrieval itself (session lookup, the empty index, k clamping, cosine distance to similarity)
runs against the real database in ``tests/integration/features/retrieval/test_retrieval_service.py``.
"""

from pathlib import Path

import pytest

from cortexdj.features.retrieval.service import (
    TrackHit,
    _participant_dat_path,
    serialize_hits,
)


class TestParticipantDatPath:
    def test_parses_standard_format(self) -> None:
        path = _participant_dat_path("P01", Path("/deap"))
        assert path == Path("/deap/s01.dat")

    def test_parses_two_digit(self) -> None:
        path = _participant_dat_path("P32", Path("/deap"))
        assert path.name == "s32.dat"

    def test_rejects_missing_prefix(self) -> None:
        with pytest.raises(ValueError, match="unexpected participant_id format"):
            _participant_dat_path("01", Path("/deap"))

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="unexpected participant_id format"):
            _participant_dat_path("", Path("/deap"))


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
