"""Spotify OAuth schemas."""

from cortexdj.schemas.base import BaseSchema


class SpotifyConnectionStatus(BaseSchema):
    """Response for Spotify connection status."""

    connected: bool
