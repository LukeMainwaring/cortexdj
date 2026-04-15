from datetime import datetime
from typing import Any

from .base import BaseSchema


class SessionSchema(BaseSchema):
    id: str
    participant_id: str
    dataset_source: str
    recorded_at: datetime
    duration_seconds: float
    metadata_extra: dict[str, Any] | None = None
    created_at: datetime


class SessionCreateSchema(BaseSchema):
    id: str
    participant_id: str
    dataset_source: str
    recorded_at: datetime
    duration_seconds: float
    metadata_extra: dict[str, Any] | None = None


class SessionListResponse(BaseSchema):
    sessions: list[SessionSchema]
    total: int


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
