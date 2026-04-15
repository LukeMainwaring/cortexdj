from __future__ import annotations

from .base import BaseSchema


class SimilarTrackSchema(BaseSchema):
    spotify_id: str
    title: str
    artist: str
    itunes_preview_url: str | None = None
    audio_cache_key: str | None = None
    similarity: float


class SimilarTracksResponse(BaseSchema):
    session_id: str
    tracks: list[SimilarTrackSchema]
    k: int
