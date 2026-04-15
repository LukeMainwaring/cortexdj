"""Pure-function tests for the audio_catalog match heuristic.

These tests pin the empirical design decisions from the 50-track probe:
the duration-delta≤3000ms hard filter and the artist-Jaccard ranking.
Regressions here will silently drop the 86% real-library hit rate, so
we'd rather catch them at pytest time.
"""

from typing import Any

from cortexdj.services.audio_catalog import (
    DURATION_DELTA_MS_HARD_LIMIT,
    _normalize,
    _pick_best,
    title_similarity,
)


class TestNormalize:
    def test_lowercase_and_trim(self) -> None:
        assert _normalize("  HELLO  ") == "hello"

    def test_strips_parenthetical_qualifiers(self) -> None:
        assert _normalize("Girls on Film (Night Version)") == "girls on film"
        assert _normalize("Creep [Explicit]") == "creep"

    def test_strips_remaster_suffix(self) -> None:
        assert _normalize("Bohemian Rhapsody - Remastered 2011") == "bohemian rhapsody"
        assert _normalize("God Only Knows - Mono") == "god only knows"

    def test_strips_feat_clause(self) -> None:
        assert _normalize("One More feat. Ad-Apt") == "one more"

    def test_strips_new_version_keywords(self) -> None:
        assert _normalize("Wonderwall - Acoustic") == "wonderwall"
        assert _normalize("Clocks - Instrumental") == "clocks"

    def test_ascii_only_after_stripping_punctuation(self) -> None:
        assert _normalize("Don't Stop Me Now!") == "dont stop me now"


class TestTitleSimilarity:
    def test_identical_titles(self) -> None:
        assert title_similarity("The Beautiful People", "The Beautiful People") == 1.0

    def test_medley_substring_trap_below_085(self) -> None:
        # The Jackson 5 bug: "Dancing Machine / Blame It on the Boogie"
        # contains the requested title as a substring but should be rejected.
        sim = title_similarity("Dancing Machine / Blame It on the Boogie", "Blame It On The Boogie")
        assert sim < 0.85

    def test_remaster_variants_identical_after_normalization(self) -> None:
        assert title_similarity("Song 2 - 2012 Remaster", "Song 2") == 1.0

    def test_unrelated_titles_low(self) -> None:
        assert title_similarity("Bohemian Rhapsody", "Imagine") == 0.0


def _mk(title: str, artist: str, dur: int) -> dict[str, Any]:
    return {"trackName": title, "artistName": artist, "trackTimeMillis": dur, "trackId": "x"}


class TestPickBest:
    def test_returns_none_when_all_exceed_duration_limit(self) -> None:
        results = [
            _mk("X", "A", 200_000 + DURATION_DELTA_MS_HARD_LIMIT + 1),
            _mk("Y", "A", 200_000 - DURATION_DELTA_MS_HARD_LIMIT - 1),
        ]
        assert _pick_best(results, artist="A", title="X", duration_ms=200_000) is None

    def test_duration_filter_accepts_exact_edge(self) -> None:
        results = [_mk("X", "A", 200_000 + DURATION_DELTA_MS_HARD_LIMIT)]
        picked = _pick_best(results, artist="A", title="X", duration_ms=200_000)
        assert picked is not None

    def test_duration_filter_rejects_green_day_9min_suite(self) -> None:
        # The Green Day bug: 9-minute "Jesus of Suburbia" suite vs radio edit.
        results = [
            _mk("Jesus of Suburbia", "Green Day", 540_000),  # 9min suite — rejected
            _mk("Jesus of Suburbia", "Green Day", 301_000),  # radio edit — accepted
        ]
        picked = _pick_best(results, artist="Green Day", title="Jesus of Suburbia", duration_ms=300_000)
        assert picked is not None and picked["trackTimeMillis"] == 301_000

    def test_ranks_higher_artist_similarity_first(self) -> None:
        results = [
            _mk("The Beautiful People", "Vitamin String Quartet", 218_500),
            _mk("The Beautiful People", "Marilyn Manson", 218_800),
        ]
        picked = _pick_best(results, artist="Marilyn Manson", title="The Beautiful People", duration_ms=218_826)
        assert picked is not None and picked["artistName"] == "Marilyn Manson"

    def test_duration_is_final_tiebreaker(self) -> None:
        # Two results with identical artist+title Jaccard; the closer duration wins.
        results = [
            _mk("Song 2", "Blur", 121_000),  # Δ=1000
            _mk("Song 2", "Blur", 120_500),  # Δ=500 — should win
            _mk("Song 2", "Blur", 122_000),  # Δ=2000
        ]
        picked = _pick_best(results, artist="Blur", title="Song 2", duration_ms=120_000)
        assert picked is not None and picked["trackTimeMillis"] == 120_500

    def test_handles_null_tracktimemillis_without_crashing(self) -> None:
        # iTunes occasionally ships null trackTimeMillis for compilations.
        results = [
            _mk("X", "A", 200_000),
            {"trackName": "X", "artistName": "A", "trackTimeMillis": None, "trackId": "y"},
        ]
        picked = _pick_best(results, artist="A", title="X", duration_ms=200_000)
        assert picked is not None and picked["trackTimeMillis"] == 200_000
