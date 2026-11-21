from datetime import datetime
from typing import Any

from cortexdj.shared.schema_base import BaseSchema


class SessionSummarySchema(BaseSchema):
    id: str
    display_index: int
    label: str
    dominant_state: str
    state_distribution: dict[str, float]
    segment_count: int
    track_count: int
    duration_seconds: float
    avg_arousal: float
    avg_valence: float


class SessionSummaryListResponse(BaseSchema):
    sessions: list[SessionSummarySchema]
    total: int


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


class BrainContext(BaseSchema):
    """The brain state a conversation is focused on; every field is optional and set independently."""

    latest_session_id: str | None = None
    dominant_mood: str | None = None
    avg_arousal: float | None = None
    avg_valence: float | None = None
