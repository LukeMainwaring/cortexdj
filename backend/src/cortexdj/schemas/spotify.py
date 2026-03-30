"""Spotify OAuth schemas."""

from datetime import datetime

from cortexdj.schemas.base import BaseSchema


class SpotifyConnectionStatus(BaseSchema):
    """Response for Spotify connection status."""

    connected: bool


class SpotifyTokenUpdate(BaseSchema):
    """Internal schema for updating stored Spotify tokens."""

    access_token: str
    refresh_token: str
    expires_at: datetime
