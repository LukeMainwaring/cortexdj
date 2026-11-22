"""Trajectory analytics for an EEG session's 2D affect path, counted per track.

The track is the unit: each ``SessionTrack`` is one point at the mean scores its segments
average to, and time in each quadrant, transitions, centroid, dispersion and path length are
computed over those points. Do not compute them over segments: segment-to-segment noise inside
a track would show up as hundreds of transitions. The result powers the Emotion Trajectory
visualization in the frontend and is included in the ``analyze_session`` tool output so the
agent can narrate the arc track by track.
"""

import logging
from collections.abc import Sequence
from math import sqrt

from cortexdj.features.sessions.emotion import EMOTION_STATES, dominant_quadrant
from cortexdj.features.sessions.model import SessionTrack
from cortexdj.features.sessions.schema import (
    SmoothedPoint,
    TrajectorySummary,
    TransitionEvent,
)

logger = logging.getLogger(__name__)


def compute_trajectory_summary(
    tracks: Sequence[SessionTrack],
    *,
    duration_seconds: float,
) -> TrajectorySummary | None:
    """Compute dwell, transitions, centroid, dispersion, path length and the path over a session's tracks.

    A session track stores no start time, so each track starts at its equal share of
    ``duration_seconds`` (every DEAP stimulus plays for the same minute); a transition's time
    is the start of the track it lands on. The path, still named ``smoothed`` so the API
    shape holds, is one unsmoothed point per track. Returns ``None`` when there are no
    tracks so callers can skip attaching the summary.
    """
    if not tracks:
        return None

    ordered = sorted(tracks, key=lambda t: t.track_order)
    n = len(ordered)
    track_seconds = duration_seconds / n
    starts = [round(i * track_seconds, 4) for i in range(n)]

    dwell: dict[str, float] = dict.fromkeys(EMOTION_STATES, 0.0)
    for t in ordered:
        if t.dominant_state in dwell:
            dwell[t.dominant_state] += 1.0
        else:
            logger.warning(f"Unknown dominant_state {t.dominant_state!r} on session track {t.id}")
    for k in dwell:
        dwell[k] = round(dwell[k] / n, 4)

    transitions = [
        TransitionEvent(time=start, from_quadrant=prev.dominant_state, to_quadrant=curr.dominant_state)
        for prev, curr, start in zip(ordered[:-1], ordered[1:], starts[1:], strict=True)
        if prev.dominant_state != curr.dominant_state
    ]

    centroid_a = sum(t.avg_arousal for t in ordered) / n
    centroid_v = sum(t.avg_valence for t in ordered) / n

    dispersion = sum(sqrt((t.avg_valence - centroid_v) ** 2 + (t.avg_arousal - centroid_a) ** 2) for t in ordered) / n

    path_length = sum(
        sqrt((b.avg_valence - a.avg_valence) ** 2 + (b.avg_arousal - a.avg_arousal) ** 2)
        for a, b in zip(ordered[:-1], ordered[1:], strict=True)
    )

    return TrajectorySummary(
        dwell_fractions=dwell,
        dominant_quadrant=dominant_quadrant(dwell),
        transition_count=len(transitions),
        transitions=transitions,
        centroid=(round(centroid_a, 4), round(centroid_v, 4)),
        dispersion=round(dispersion, 4),
        path_length=round(path_length, 4),
        smoothed=[
            SmoothedPoint(
                start_time=start,
                arousal=round(t.avg_arousal, 4),
                valence=round(t.avg_valence, 4),
                quadrant=t.dominant_state,
            )
            for t, start in zip(ordered, starts, strict=True)
        ],
    )
