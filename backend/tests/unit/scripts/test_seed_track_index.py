"""The candidate pool ``seed-track-index`` builds: dedupe, library-then-seed top-up, and the limit.

``_gather_candidates`` runs against a hand-written fake spotipy client passed in, so what a
test observes is the returned pool and which Spotify calls were made.
"""

from typing import Any, cast

import spotipy

from cortexdj.scripts.seed_track_index import _dedupe_by_spotify_id, _gather_candidates


def _candidate(spotify_id: str, source: str = "user_library") -> dict[str, Any]:
    return {
        "spotify_id": spotify_id,
        "title": f"T{spotify_id}",
        "artist": f"A{spotify_id}",
        "duration_ms": 200_000,
        "source": source,
    }


def _track(spotify_id: str) -> dict[str, Any]:
    return {"id": spotify_id, "name": f"T{spotify_id}", "artists": [{"name": f"A{spotify_id}"}], "duration_ms": 200_000}


class FakeSpotify:
    """The two spotipy calls the seeder makes: a paged saved-tracks library, and a search per genre seed.

    A search returns ``search_ids`` when given, the same for every seed; otherwise ids of
    its own named after the seed (``"ambient:0"``, ``"ambient:1"``, ...). Either is cut to
    the requested limit.
    """

    def __init__(self, library_ids: list[str], search_ids: list[str] | None = None) -> None:
        self.library_ids = library_ids
        self.search_ids = search_ids or []
        self.library_calls: list[tuple[int, int]] = []
        self.searches: list[str] = []

    def current_user_saved_tracks(self, limit: int, offset: int) -> dict[str, Any]:
        self.library_calls.append((limit, offset))
        return {"items": [{"track": _track(i)} for i in self.library_ids[offset : offset + limit]]}

    def search(self, q: str, type: str, limit: int) -> dict[str, Any]:
        self.searches.append(q)
        ids = self.search_ids or [f"{q}:{i}" for i in range(limit)]
        return {"tracks": {"items": [_track(i) for i in ids[:limit]]}}


def _client(fake: FakeSpotify) -> spotipy.Spotify:
    return cast(spotipy.Spotify, fake)


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
    async def test_a_library_that_fills_the_limit_never_searches(self) -> None:
        fake = FakeSpotify(library_ids=[f"lib{i}" for i in range(10)], search_ids=["seed0"])

        pool = await _gather_candidates(_client(fake), limit=10)

        assert [c["spotify_id"] for c in pool] == [f"lib{i}" for i in range(10)]
        assert {c["source"] for c in pool} == {"user_library"}
        assert fake.searches == []

    async def test_a_library_shortfall_is_topped_up_from_the_seeds(self) -> None:
        fake = FakeSpotify(library_ids=["lib0", "lib1", "lib2"])

        pool = await _gather_candidates(_client(fake), limit=10)

        assert [c["spotify_id"] for c in pool] == [
            "lib0",
            "lib1",
            "lib2",
            "ambient:0",
            "ambient:1",
            "ambient:2",
            "ambient:3",
            "ambient:4",
            "chillhop:0",
            "chillhop:1",
        ]
        assert [c["source"] for c in pool] == ["user_library"] * 3 + ["seed_search"] * 7
        # The shortfall of 7 is overshot to 14 so seed-side dedupe losses still fill the pool;
        # at the floor of 5 per seed that takes three searches.
        assert fake.searches == ["ambient", "chillhop", "indie rock"]

    async def test_seed_duplicates_of_library_tracks_are_dropped(self) -> None:
        lib_ids = ["a", "b", "c", "d", "e"]
        fake = FakeSpotify(library_ids=lib_ids, search_ids=["a", "b", "x", "y", "z"])

        pool = await _gather_candidates(_client(fake), limit=10)

        assert [c["spotify_id"] for c in pool] == ["a", "b", "c", "d", "e", "x", "y", "z"]
        # The library copy is kept, not the seed duplicate.
        assert [c["source"] for c in pool[:5]] == ["user_library"] * 5

    async def test_the_pool_is_cut_to_the_limit(self) -> None:
        fake = FakeSpotify(library_ids=["lib0", "lib1"])

        pool = await _gather_candidates(_client(fake), limit=4)

        # Seeds were asked for 2 × the shortfall of 2; the overshoot is cut off.
        assert [c["spotify_id"] for c in pool] == ["lib0", "lib1", "ambient:0", "ambient:1"]
