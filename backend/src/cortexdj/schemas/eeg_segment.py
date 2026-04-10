from datetime import datetime
from typing import Any

from .base import BaseSchema


class BandPowers(BaseSchema):
    delta: float
    theta: float
    alpha: float
    beta: float
    gamma: float


class SegmentSchema(BaseSchema):
    id: str
    session_id: str
    segment_index: int
    start_time: float
    end_time: float
    arousal_score: float
    valence_score: float
    dominant_state: str
    band_powers: dict[str, float]
    features: dict[str, Any] | None = None
    created_at: datetime


class TransitionEvent(BaseSchema):
    time: float
    from_quadrant: str
    to_quadrant: str


class SmoothedPoint(BaseSchema):
    start_time: float
    arousal: float
    valence: float
    quadrant: str


class TrajectorySummary(BaseSchema):
    dwell_fractions: dict[str, float]
    dominant_quadrant: str
    transition_count: int
    transitions: list[TransitionEvent]
    centroid: tuple[float, float]
    dispersion: float
    path_length: float
    smoothed: list[SmoothedPoint]


class SegmentListResponse(BaseSchema):
    segments: list[SegmentSchema]
    total: int
    trajectory_summary: TrajectorySummary | None = None
