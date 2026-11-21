"""Spotify-slice fixtures for the integration tier: every test here runs with Spotify unconfigured.

Unconfigured settings make the client getter's answer depend only on the ``spotify_tokens``
row, and make a token refresh fail before it reaches the network (spotipy refuses an OAuth
manager without a client id). spotipy also reads ``SPOTIPY_CLIENT_ID`` from the environment
as a fallback, so that goes too.

Do not add a fixture that configures Spotify: ``client.py`` caches the app client for the
process once settings are configured, so every later test would get a real client that
calls Spotify.
"""

from collections.abc import Iterator

import pytest

from cortexdj.core.config import get_settings


@pytest.fixture(autouse=True)
def _spotify_unconfigured(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET", "SPOTIPY_CLIENT_ID", "SPOTIPY_CLIENT_SECRET"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
