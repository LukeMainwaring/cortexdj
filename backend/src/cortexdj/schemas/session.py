"""EEG session schemas."""

from datetime import datetime
from typing import Any

from .base import BaseSchema


class SessionSchema(BaseSchema):
    """Full session representation."""

    id: str
    participant_id: str
    dataset_source: str
    recorded_at: datetime
    duration_seconds: float
    metadata_extra: dict[str, Any] | None = None
    created_at: datetime


class SessionCreateSchema(BaseSchema):
    """Schema for creating a new session."""

    id: str
    participant_id: str
    dataset_source: str
    recorded_at: datetime
    duration_seconds: float
    metadata_extra: dict[str, Any] | None = None


class SessionListResponse(BaseSchema):
    """Response containing list of sessions."""

    sessions: list[SessionSchema]
    total: int
