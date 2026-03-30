"""Track schemas."""

from datetime import datetime
from typing import Any

from .base import BaseSchema


class TrackSchema(BaseSchema):
    """Full track representation."""

    id: str
    spotify_track_id: str | None = None
    title: str
    artist: str
    album: str | None = None
    duration_ms: int | None = None
    spotify_features: dict[str, Any] | None = None
    created_at: datetime


class TrackCreateSchema(BaseSchema):
    """Schema for creating a new track."""

    id: str
    spotify_track_id: str | None = None
    title: str
    artist: str
    album: str | None = None
    duration_ms: int | None = None
