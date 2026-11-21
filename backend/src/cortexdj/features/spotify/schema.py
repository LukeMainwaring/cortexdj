from datetime import datetime
from typing import Any

from cortexdj.shared.schema_base import BaseSchema


class SpotifyConnectionStatus(BaseSchema):
    connected: bool


class PlaylistSchema(BaseSchema):
    id: str
    spotify_playlist_id: str | None = None
    name: str
    mood_criteria: dict[str, Any] | None = None
    track_count: int
    created_at: datetime


class PlaylistCreateSchema(BaseSchema):
    id: str
    spotify_playlist_id: str | None = None
    name: str
    mood_criteria: dict[str, Any] | None = None
    track_count: int = 0
