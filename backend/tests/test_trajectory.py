"""Tests for the trajectory analytics service."""

from __future__ import annotations

from datetime import UTC, datetime
from math import isclose, sqrt

from cortexdj.ml.dataset import scores_to_quadrant
from cortexdj.models.eeg_segment import EegSegment
from cortexdj.services.trajectory import (
    compute_trajectory_summary,
    quadrant_from_scores,
    smooth_trajectory,
)


def _segment(
    index: int, arousal: float, valence: float, state: str, start: float
) -> EegSegment:
    return EegSegment(
        id=f"seg-{index}",
        session_id="s",
        segment_index=index,
        start_time=start,
        end_time=start + 4.0,
        arousal_score=arousal,
        valence_score=valence,
        dominant_state=state,
        band_powers={"delta": 0.0, "theta": 0.0, "alpha": 0.0, "beta": 0.0, "gamma": 0.0},
        features=None,
        created_at=datetime.now(tz=UTC),
    )


class TestQuadrantFromScores:
    def test_parity_with_scores_to_quadrant(self) -> None:
        # quadrant_from_scores takes 0-1 inputs and must agree with the 0-10
        # canonical mapping used by ml.dataset and ml.predict.
        cases = [(0.2, 0.2), (0.2, 0.8), (0.8, 0.2), (0.8, 0.8), (0.5, 0.5)]
        for a, v in cases:
            assert quadrant_from_scores(a, v) == scores_to_quadrant(a * 10, v * 10)


class TestSmoothTrajectory:
    def test_empty(self) -> None:
        assert smooth_trajectory([]) == []

    def test_window_3_rolling_mean(self) -> None:
        segs = [
            _segment(0, 0.10, 0.10, "calm", 0.0),
            _segment(1, 0.30, 0.30, "calm", 4.0),
            _segment(2, 0.50, 0.50, "excited", 8.0),
            _segment(3, 0.70, 0.70, "excited", 12.0),
            _segment(4, 0.90, 0.90, "excited", 16.0),
        ]
        points = smooth_trajectory(segs, window=3)

        # Endpoints use a 2-wide half-window; interior uses full 3-wide.
        assert isclose(points[0].arousal, (0.10 + 0.30) / 2, abs_tol=1e-4)
        assert isclose(points[1].arousal, (0.10 + 0.30 + 0.50) / 3, abs_tol=1e-4)
        assert isclose(points[2].valence, (0.30 + 0.50 + 0.70) / 3, abs_tol=1e-4)
        assert isclose(points[4].valence, (0.70 + 0.90) / 2, abs_tol=1e-4)
        assert [p.start_time for p in points] == [s.start_time for s in segs]


class TestComputeTrajectorySummary:
    def test_empty_returns_none(self) -> None:
        assert compute_trajectory_summary([]) is None

    def test_dwell_transitions_and_metrics(self) -> None:
        segs = [
            _segment(0, 0.2, 0.2, "calm", 0.0),
            _segment(1, 0.2, 0.2, "calm", 4.0),
            _segment(2, 0.8, 0.8, "excited", 8.0),
            _segment(3, 0.2, 0.8, "relaxed", 12.0),
        ]

        summary = compute_trajectory_summary(segs)
        assert summary is not None

        assert summary.dwell_fractions["calm"] == 0.5
        assert summary.dwell_fractions["excited"] == 0.25
        assert summary.dwell_fractions["relaxed"] == 0.25
        assert summary.dwell_fractions["stressed"] == 0.0

        assert summary.dominant_quadrant == "calm"

        assert summary.transition_count == 2
        assert summary.transitions[0].from_quadrant == "calm"
        assert summary.transitions[0].to_quadrant == "excited"
        assert summary.transitions[0].time == 8.0
        assert summary.transitions[1].from_quadrant == "excited"
        assert summary.transitions[1].to_quadrant == "relaxed"

        expected_vc = (0.2 + 0.2 + 0.8 + 0.8) / 4
        expected_ac = (0.2 + 0.2 + 0.8 + 0.2) / 4
        assert isclose(summary.centroid[0], round(expected_vc, 4))
        assert isclose(summary.centroid[1], round(expected_ac, 4))

        # Path length: three edges between four points in (valence, arousal)
        expected_path = (
            sqrt((0.2 - 0.2) ** 2 + (0.2 - 0.2) ** 2)
            + sqrt((0.8 - 0.2) ** 2 + (0.8 - 0.2) ** 2)
            + sqrt((0.8 - 0.8) ** 2 + (0.2 - 0.8) ** 2)
        )
        assert isclose(summary.path_length, round(expected_path, 4), abs_tol=1e-4)

        assert summary.dispersion > 0.0
        assert len(summary.smoothed) == len(segs)

    def test_single_segment_has_zero_path_and_no_transitions(self) -> None:
        segs = [_segment(0, 0.5, 0.5, "excited", 0.0)]
        summary = compute_trajectory_summary(segs)
        assert summary is not None
        assert summary.path_length == 0.0
        assert summary.transition_count == 0
        assert summary.dispersion == 0.0
        assert summary.dominant_quadrant == "excited"
