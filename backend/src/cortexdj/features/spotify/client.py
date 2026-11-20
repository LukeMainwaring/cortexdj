"""The one place the slice decides which Spotify client a tool or tool preparation uses.

The user's own client when a token row exists (refreshed when it is about to expire),
else the app's client-credentials client, else no client when Spotify is not configured.
The result says which it is, because the user-only tools are offered and a mood playlist
is created on Spotify only with the user's own login.

Resolved from the database session on every call, not carried on ``AgentDeps``: the user
can connect or disconnect between runs, and ``AgentDeps`` stays the single field ``db``.
That costs one token-row read per model step. Do not cache the user's client across calls.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.core.config import get_settings
from cortexdj.features.spotify.model import SpotifyToken
from cortexdj.features.spotify.service import SPOTIFY_SCOPES, run_spotify

logger = logging.getLogger(__name__)

# A token this close to expiry is refreshed before use, so a run never starts on one
# that lapses mid-call.
_REFRESH_BUFFER = timedelta(minutes=5)


@dataclass(frozen=True)
class SpotifyClient:
    """The client to call Spotify with; ``client`` is None when Spotify is not configured.

    ``is_user`` is True only for the user's own login, the client that may read their
    library and create playlists on their account.
    """

    client: spotipy.Spotify | None
    is_user: bool


_app_client: spotipy.Spotify | None = None
_override: SpotifyClient | None = None


def _get_app_client() -> spotipy.Spotify | None:
    """Returns None if Spotify credentials are not configured."""
    global _app_client
    settings = get_settings()
    if _app_client is None and settings.spotify_configured:
        _app_client = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=settings.SPOTIFY_CLIENT_ID,
                client_secret=settings.SPOTIFY_CLIENT_SECRET,
            )
        )
    return _app_client


async def _get_user_client(db: AsyncSession) -> spotipy.Spotify | None:
    """Get a user-authenticated Spotify client from stored OAuth tokens.

    Auto-refreshes if token is expired or expires within 5 minutes.
    Returns None if no tokens are stored (user hasn't connected Spotify), or if a
    refresh fails: the caller then falls back to the app client.
    """
    token = await SpotifyToken.get(db)
    if token is None:
        return None

    needs_refresh = token.expires_at <= datetime.now(UTC) + _REFRESH_BUFFER

    access_token = token.access_token

    if needs_refresh and token.refresh_token:
        logger.info("Refreshing Spotify access token")
        settings = get_settings()
        try:
            oauth = SpotifyOAuth(
                client_id=settings.SPOTIFY_CLIENT_ID,
                client_secret=settings.SPOTIFY_CLIENT_SECRET,
                redirect_uri=settings.SPOTIFY_REDIRECT_URI,
                scope=SPOTIFY_SCOPES,
            )
            token_info = await run_spotify(oauth.refresh_access_token, token.refresh_token)

            if token_info:
                expires_at = datetime.fromtimestamp(token_info["expires_at"], tz=UTC)
                await SpotifyToken.upsert(
                    db,
                    access_token=token_info["access_token"],
                    refresh_token=token_info.get("refresh_token", token.refresh_token),
                    expires_at=expires_at,
                )
                access_token = token_info["access_token"]
                logger.info("Successfully refreshed Spotify access token")
        except Exception:  # a failed refresh means "not connected", not a failed run
            logger.warning("Failed to refresh Spotify token, falling back to Client Credentials")
            return None

    return spotipy.Spotify(auth=access_token)


async def get_spotify_client(db: AsyncSession) -> SpotifyClient:
    """The user's client if connected, else the app client (``client`` None when unconfigured).

    Under ``override_spotify_client`` the override is returned without touching the database.
    """
    if _override is not None:
        return _override
    user_client = await _get_user_client(db)
    if user_client is not None:
        return SpotifyClient(client=user_client, is_user=True)
    return SpotifyClient(client=_get_app_client(), is_user=False)


@contextmanager
def override_spotify_client(spotify_client: SpotifyClient) -> Iterator[None]:
    """Make ``get_spotify_client`` return ``spotify_client`` for the length of a ``with`` block.

    For tests: a fake user client (``is_user=True``), the app client, or
    ``SpotifyClient(client=None, is_user=False)`` to stand for "not configured". The app
    never calls this.
    """
    global _override
    previous = _override
    _override = spotify_client
    try:
        yield
    finally:
        _override = previous
