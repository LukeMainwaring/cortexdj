"""Tests for prediction utilities: quadrant mapping and label thresholds."""

from cortexdj.ml.dataset import (
    AROUSAL_THRESHOLD,
    VALENCE_THRESHOLD,
    scores_to_quadrant,
)


class TestScoresToQuadrant:
    def test_high_arousal_high_valence_is_excited(self) -> None:
        assert scores_to_quadrant(7.0, 7.0) == "excited"

    def test_low_arousal_high_valence_is_relaxed(self) -> None:
        assert scores_to_quadrant(3.0, 7.0) == "relaxed"

    def test_high_arousal_low_valence_is_stressed(self) -> None:
        assert scores_to_quadrant(7.0, 3.0) == "stressed"

    def test_low_arousal_low_valence_is_calm(self) -> None:
        assert scores_to_quadrant(3.0, 3.0) == "calm"

    def test_boundary_values_at_threshold(self) -> None:
        # At exactly the threshold, both are >= so it's "excited"
        assert scores_to_quadrant(AROUSAL_THRESHOLD, VALENCE_THRESHOLD) == "excited"

    def test_just_below_arousal_threshold(self) -> None:
        assert scores_to_quadrant(AROUSAL_THRESHOLD - 0.01, 7.0) == "relaxed"

    def test_just_below_valence_threshold(self) -> None:
        assert scores_to_quadrant(7.0, VALENCE_THRESHOLD - 0.01) == "stressed"

    def test_extreme_values(self) -> None:
        assert scores_to_quadrant(10.0, 10.0) == "excited"
        assert scores_to_quadrant(0.0, 0.0) == "calm"
