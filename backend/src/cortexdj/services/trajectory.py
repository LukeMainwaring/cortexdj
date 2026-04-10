"""Trajectory analytics for an EEG session's 2D affect path.

Given a time-ordered sequence of ``EegSegment`` rows, compute dwell time per
emotion quadrant, transitions, centroid, dispersion, total path length, and a
rolling-mean smoothed trail. The result powers the Emotion Trajectory
visualization in the frontend and is included in the ``analyze_session`` tool
output so the agent can narrate the emotional arc.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

from cortexdj.ml.dataset import scores_to_quadrant
from cortexdj.models.eeg_segment import EegSegment
from cortexdj.schemas.eeg_segment import (
    SmoothedPoint,
    TrajectorySummary,
    TransitionEvent,
)

EMOTION_STATES: tuple[str, ...] = ("relaxed", "calm", "excited", "stressed")


def quadrant_from_scores(arousal: float, valence: float) -> str:
    """Map arousal/valence in ``[0, 1]`` to a canonical emotion quadrant.

    Thin wrapper around :func:`cortexdj.ml.dataset.scores_to_quadrant`, which
    expects 0-10 inputs. Kept here so the trajectory service owns a single
    quadrant-mapping surface.
    """
    return scores_to_quadrant(arousal * 10, valence * 10)


def smooth_trajectory(
    segments: Sequence[EegSegment], window: int = 3
) -> list[SmoothedPoint]:
    """Rolling-mean arousal/valence across a time-ordered segment list."""
    if not segments:
        return []
    half = window // 2
    points: list[SmoothedPoint] = []
    for i, seg in enumerate(segments):
        lo = max(0, i - half)
        hi = min(len(segments), i + half + 1)
        window_segs = segments[lo:hi]
        avg_a = sum(s.arousal_score for s in window_segs) / len(window_segs)
        avg_v = sum(s.valence_score for s in window_segs) / len(window_segs)
        points.append(
            SmoothedPoint(
                start_time=seg.start_time,
                arousal=round(avg_a, 4),
                valence=round(avg_v, 4),
                quadrant=quadrant_from_scores(avg_a, avg_v),
            )
        )
    return points


def compute_trajectory_summary(
    segments: Sequence[EegSegment],
) -> TrajectorySummary | None:
    """Compute dwell, transitions, centroid, dispersion, path length, smoothed trail.

    Returns ``None`` when there are no segments so callers can skip attaching
    the summary rather than serializing empty fields.
    """
    if not segments:
        return None

    n = len(segments)

    dwell: dict[str, float] = dict.fromkeys(EMOTION_STATES, 0.0)
    for s in segments:
        dwell[s.dominant_state] = dwell.get(s.dominant_state, 0.0) + 1.0
    for k in dwell:
        dwell[k] = round(dwell[k] / n, 4)

    dominant_quadrant = max(dwell, key=lambda k: dwell[k])

    transitions: list[TransitionEvent] = []
    for prev, curr in zip(segments[:-1], segments[1:]):
        if prev.dominant_state != curr.dominant_state:
            transitions.append(
                TransitionEvent(
                    time=round(curr.start_time, 4),
                    from_quadrant=prev.dominant_state,
                    to_quadrant=curr.dominant_state,
                )
            )

    centroid_v = sum(s.valence_score for s in segments) / n
    centroid_a = sum(s.arousal_score for s in segments) / n

    dispersion = (
        sum(
            sqrt(
                (s.valence_score - centroid_v) ** 2
                + (s.arousal_score - centroid_a) ** 2
            )
            for s in segments
        )
        / n
    )

    path_length = sum(
        sqrt(
            (b.valence_score - a.valence_score) ** 2
            + (b.arousal_score - a.arousal_score) ** 2
        )
        for a, b in zip(segments[:-1], segments[1:])
    )

    return TrajectorySummary(
        dwell_fractions=dwell,
        dominant_quadrant=dominant_quadrant,
        transition_count=len(transitions),
        transitions=transitions,
        centroid=(round(centroid_v, 4), round(centroid_a, 4)),
        dispersion=round(dispersion, 4),
        path_length=round(path_length, 4),
        smoothed=smooth_trajectory(segments),
    )
