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


class SegmentListResponse(BaseSchema):
    segments: list[SegmentSchema]
    total: int
