"""Unit tests for scripts.build_track_index pagination and topup logic.

The pagination / dedupe / topup math in `_gather_candidates` is the only
non-trivial pure-ish logic in the seeder script. The review flagged this
section as buggy at first revision — these tests pin the corrected
behavior so a future "optimization" can't silently regress it.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from cortexdj.scripts.build_track_index import _dedupe_by_spotify_id, _gather_candidates


def _candidate(spotify_id: str, source: str = "user_library") -> dict[str, Any]:
    return {
        "spotify_id": spotify_id,
        "title": f"T{spotify_id}",
        "artist": f"A{spotify_id}",
        "duration_ms": 200_000,
        "source": source,
    }


class TestDedupeBySpotifyId:
    def test_empty(self) -> None:
        assert _dedupe_by_spotify_id([]) == []

    def test_preserves_order_of_first_occurrence(self) -> None:
        result = _dedupe_by_spotify_id([_candidate("a"), _candidate("b"), _candidate("a")])
        assert [c["spotify_id"] for c in result] == ["a", "b"]

    def test_same_id_different_source_dedupes(self) -> None:
        lib = _candidate("a", source="user_library")
        seed = _candidate("a", source="seed_search")
        result = _dedupe_by_spotify_id([lib, seed])
        assert len(result) == 1
        # First occurrence wins — library is kept over the seed duplicate.
        assert result[0]["source"] == "user_library"

    def test_all_unique(self) -> None:
        result = _dedupe_by_spotify_id([_candidate(s) for s in ("a", "b", "c")])
        assert len(result) == 3


class TestGatherCandidates:
    def test_library_only_when_full(self) -> None:
        # Library returns exactly `limit` candidates — topup should be skipped.
        client = MagicMock()
        with (
            patch(
                "cortexdj.scripts.build_track_index._fetch_saved_track_candidates",
                new=AsyncMock(return_value=[_candidate(str(i)) for i in range(10)]),
            ),
            patch(
                "cortexdj.scripts.build_track_index._fetch_genre_seed_candidates",
                new=AsyncMock(return_value=[]),
            ) as mock_seeds,
        ):
            pool = asyncio.run(_gather_candidates(client, limit=10, skip_library=False))
        assert len(pool) == 10
        mock_seeds.assert_not_called()

    def test_library_shortfall_triggers_seed_topup(self) -> None:
        # Library returns 3, we need 10 — seeds must top up the remaining 7.
        # The corrected implementation overshoots with `shortfall * 2 = 14`
        # to absorb seed-side dedupe losses.
        client = MagicMock()
        with (
            patch(
                "cortexdj.scripts.build_track_index._fetch_saved_track_candidates",
                new=AsyncMock(return_value=[_candidate(f"lib{i}") for i in range(3)]),
            ),
            patch(
                "cortexdj.scripts.build_track_index._fetch_genre_seed_candidates",
                new=AsyncMock(return_value=[_candidate(f"seed{i}", source="seed_search") for i in range(14)]),
            ) as mock_seeds,
        ):
            pool = asyncio.run(_gather_candidates(client, limit=10, skip_library=False))
        assert len(pool) == 10
        # Seeds should have been asked for shortfall * 2 = 14.
        mock_seeds.assert_called_once()
        assert mock_seeds.call_args.kwargs["max_tracks"] == 14
        # The 3 library candidates should be first (source preserved).
        assert [c["source"] for c in pool[:3]] == ["user_library"] * 3

    def test_library_and_seed_dedupe(self) -> None:
        # Library returns 5 unique tracks; seeds return 5 that overlap with
        # library + 3 fresh ones. After dedupe the pool should contain
        # 5 library + 3 fresh seed = 8, sliced to limit=10 → all 8.
        client = MagicMock()
        lib_ids = ["a", "b", "c", "d", "e"]
        seed_ids = ["a", "b", "c", "d", "e", "x", "y", "z"]
        with (
            patch(
                "cortexdj.scripts.build_track_index._fetch_saved_track_candidates",
                new=AsyncMock(return_value=[_candidate(i) for i in lib_ids]),
            ),
            patch(
                "cortexdj.scripts.build_track_index._fetch_genre_seed_candidates",
                new=AsyncMock(return_value=[_candidate(i, source="seed_search") for i in seed_ids]),
            ),
        ):
            pool = asyncio.run(_gather_candidates(client, limit=10, skip_library=False))
        spotify_ids = [c["spotify_id"] for c in pool]
        assert len(spotify_ids) == 8
        # Library order preserved, fresh seeds appended.
        assert spotify_ids[:5] == lib_ids
        assert set(spotify_ids[5:]) == {"x", "y", "z"}
        # Dedupe kept the library copy, not the seed duplicate.
        assert all(c["source"] == "user_library" for c in pool[:5])

    def test_skip_library_uses_only_seeds(self) -> None:
        client = MagicMock()
        with (
            patch(
                "cortexdj.scripts.build_track_index._fetch_saved_track_candidates",
            ) as mock_lib,
            patch(
                "cortexdj.scripts.build_track_index._fetch_genre_seed_candidates",
                new=AsyncMock(return_value=[_candidate(f"seed{i}", source="seed_search") for i in range(20)]),
            ),
        ):
            pool = asyncio.run(_gather_candidates(client, limit=10, skip_library=True))
        mock_lib.assert_not_called()
        assert len(pool) == 10
        assert all(c["source"] == "seed_search" for c in pool)

    def test_slice_to_limit(self) -> None:
        # Library + seeds far exceed limit — final pool must be sliced down.
        client = MagicMock()
        with (
            patch(
                "cortexdj.scripts.build_track_index._fetch_saved_track_candidates",
                new=AsyncMock(return_value=[_candidate(f"lib{i}") for i in range(50)]),
            ),
            patch(
                "cortexdj.scripts.build_track_index._fetch_genre_seed_candidates",
                new=AsyncMock(return_value=[_candidate(f"seed{i}", source="seed_search") for i in range(50)]),
            ) as mock_seeds,
        ):
            pool = asyncio.run(_gather_candidates(client, limit=10, skip_library=False))
        # Library already fills the limit — seeds never fetched.
        assert len(pool) == 10
        mock_seeds.assert_not_called()
