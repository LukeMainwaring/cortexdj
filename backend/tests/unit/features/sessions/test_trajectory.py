"""Tests for the trajectory analytics, computed over a session's tracks."""

from math import isclose, sqrt

from cortexdj.features.sessions.model import SessionTrack
from cortexdj.features.sessions.trajectory import compute_trajectory_summary
from tests.factories import make_session_track


def _tracks(points: list[tuple[float, float, str]]) -> list[SessionTrack]:
    return [
        make_session_track("s", f"t{i}", track_order=i, avg_arousal=arousal, avg_valence=valence, dominant_state=state)
        for i, (arousal, valence, state) in enumerate(points)
    ]


class TestComputeTrajectorySummary:
    def test_no_tracks_returns_none(self) -> None:
        assert compute_trajectory_summary([], duration_seconds=0.0) is None

    def test_dwell_transitions_and_metrics_over_tracks(self) -> None:
        tracks = _tracks(
            [
                (0.2, 0.2, "calm"),
                (0.2, 0.2, "calm"),
                (0.8, 0.8, "excited"),
                (0.2, 0.8, "relaxed"),
            ]
        )

        # Four one-minute tracks.
        summary = compute_trajectory_summary(tracks, duration_seconds=240.0)
        assert summary is not None

        assert summary.dwell_fractions == {"relaxed": 0.25, "calm": 0.5, "excited": 0.25, "stressed": 0.0}
        assert summary.dominant_quadrant == "calm"

        assert summary.transition_count == 2
        assert [(t.from_quadrant, t.to_quadrant, t.time) for t in summary.transitions] == [
            ("calm", "excited", 120.0),
            ("excited", "relaxed", 180.0),
        ]

        # Centroid is (arousal, valence) to match the codebase-wide ordering.
        assert summary.centroid == (0.35, 0.5)
        expected_path = sqrt(0.6**2 + 0.6**2) + 0.6
        assert isclose(summary.path_length, round(expected_path, 4), abs_tol=1e-4)
        assert summary.dispersion > 0.0

    def test_path_is_one_unsmoothed_point_per_track(self) -> None:
        tracks = _tracks([(0.1, 0.1, "calm"), (0.9, 0.9, "excited"), (0.1, 0.1, "calm")])

        summary = compute_trajectory_summary(tracks, duration_seconds=180.0)
        assert summary is not None

        assert [(p.start_time, p.arousal, p.valence, p.quadrant) for p in summary.smoothed] == [
            (0.0, 0.1, 0.1, "calm"),
            (60.0, 0.9, 0.9, "excited"),
            (120.0, 0.1, 0.1, "calm"),
        ]

    def test_path_follows_track_order(self) -> None:
        tracks = list(reversed(_tracks([(0.1, 0.1, "calm"), (0.9, 0.9, "excited")])))

        summary = compute_trajectory_summary(tracks, duration_seconds=120.0)
        assert summary is not None

        assert [p.quadrant for p in summary.smoothed] == ["calm", "excited"]
        assert [(t.from_quadrant, t.to_quadrant, t.time) for t in summary.transitions] == [("calm", "excited", 60.0)]

    def test_tracks_in_the_same_quadrant_make_no_transition(self) -> None:
        # Whatever the segments inside these tracks did, each track has one quadrant.
        tracks = _tracks([(0.7, 0.6, "excited"), (0.8, 0.7, "excited"), (0.6, 0.9, "excited")])

        summary = compute_trajectory_summary(tracks, duration_seconds=180.0)
        assert summary is not None

        assert summary.transition_count == 0
        assert summary.transitions == []
        assert summary.dwell_fractions["excited"] == 1.0

    def test_single_track_has_zero_path_and_no_transitions(self) -> None:
        tracks = _tracks([(0.5, 0.5, "excited")])
        summary = compute_trajectory_summary(tracks, duration_seconds=60.0)
        assert summary is not None
        assert summary.path_length == 0.0
        assert summary.transition_count == 0
        assert summary.dispersion == 0.0
        assert summary.dominant_quadrant == "excited"
