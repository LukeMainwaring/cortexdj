"""Tests for the trajectory analytics service."""

from math import isclose, sqrt

from cortexdj.features.sessions.trajectory import (
    compute_trajectory_summary,
    smooth_trajectory,
)
from tests.factories import make_eeg_segment


class TestSmoothTrajectory:
    def test_empty(self) -> None:
        assert smooth_trajectory([]) == []

    def test_window_3_rolling_mean(self) -> None:
        segs = [
            make_eeg_segment(
                "s",
                segment_index=i,
                start_time=4.0 * i,
                end_time=4.0 * i + 4.0,
                arousal_score=arousal,
                valence_score=valence,
                dominant_state=state,
            )
            for i, (arousal, valence, state) in enumerate(
                [
                    (0.10, 0.10, "calm"),
                    (0.30, 0.30, "calm"),
                    (0.50, 0.50, "excited"),
                    (0.70, 0.70, "excited"),
                    (0.90, 0.90, "excited"),
                ]
            )
        ]
        points = smooth_trajectory(segs)

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
            make_eeg_segment(
                "s",
                segment_index=i,
                start_time=4.0 * i,
                end_time=4.0 * i + 4.0,
                arousal_score=arousal,
                valence_score=valence,
                dominant_state=state,
            )
            for i, (arousal, valence, state) in enumerate(
                [
                    (0.2, 0.2, "calm"),
                    (0.2, 0.2, "calm"),
                    (0.8, 0.8, "excited"),
                    (0.2, 0.8, "relaxed"),
                ]
            )
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

        expected_ac = (0.2 + 0.2 + 0.8 + 0.2) / 4
        expected_vc = (0.2 + 0.2 + 0.8 + 0.8) / 4
        # Centroid is (arousal, valence) to match the codebase-wide ordering.
        assert isclose(summary.centroid[0], round(expected_ac, 4))
        assert isclose(summary.centroid[1], round(expected_vc, 4))

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
        segs = [make_eeg_segment("s", start_time=0.0, arousal_score=0.5, valence_score=0.5, dominant_state="excited")]
        summary = compute_trajectory_summary(segs)
        assert summary is not None
        assert summary.path_length == 0.0
        assert summary.transition_count == 0
        assert summary.dispersion == 0.0
        assert summary.dominant_quadrant == "excited"
