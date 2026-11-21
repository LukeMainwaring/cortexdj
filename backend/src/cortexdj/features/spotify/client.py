"""The one place the slice gets a Spotify client: the user's own login, or none.

Every Spotify call the app makes runs on the user's login, the ``spotify_tokens`` row
(refreshed when it is about to expire). With no row, or a refresh that fails, there is
no client: the capability hides the Spotify tools and a script stops. There is no
app-level client-credentials client; do not add one as a fallback, a second kind of
client is what the tools would then have to tell apart.

Resolved from the database session on every call, not carried on ``AgentDeps``: the user
can connect or disconnect between runs, and ``AgentDeps`` stays the single field ``db``.
That costs one token-row read per model step. Do not cache the client across calls.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.core.config import get_settings
from cortexdj.features.spotify.model import SpotifyToken
from cortexdj.features.spotify.service import SPOTIFY_SCOPES, run_spotify

logger = logging.getLogger(__name__)

# A token this close to expiry is refreshed before use, so a run never starts on one
# that lapses mid-call.
_REFRESH_BUFFER = timedelta(minutes=5)


# The innermost ``override_spotify_client`` is last; empty outside tests. A stack and not
# one slot because None is itself an override ("not connected").
_overrides: list[spotipy.Spotify | None] = []


async def get_spotify_client(db: AsyncSession) -> spotipy.Spotify | None:
    """The user's Spotify client, or None when no login is stored or its refresh fails.

    Under ``override_spotify_client`` the override is returned without touching the database.
    """
    if _overrides:
        return _overrides[-1]

    token = await SpotifyToken.get(db)
    if token is None:
        return None

    access_token = token.access_token

    if token.expires_at <= datetime.now(UTC) + _REFRESH_BUFFER:
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
            await SpotifyToken.upsert(
                db,
                access_token=token_info["access_token"],
                refresh_token=token_info["refresh_token"],
                expires_at=datetime.fromtimestamp(token_info["expires_at"], tz=UTC),
            )
            access_token = token_info["access_token"]
            logger.info("Successfully refreshed Spotify access token")
        except Exception:  # a failed refresh means "not connected", not a failed run
            logger.warning("Failed to refresh Spotify token; treating Spotify as not connected")
            return None

    return spotipy.Spotify(auth=access_token)


@contextmanager
def override_spotify_client(client: spotipy.Spotify | None) -> Iterator[None]:
    """Make ``get_spotify_client`` return ``client`` for the length of a ``with`` block.

    For tests: a fake spotipy client stands for a connected user, None for not connected.
    The app never calls this.
    """
    _overrides.append(client)
    try:
        yield
    finally:
        _overrides.pop()
