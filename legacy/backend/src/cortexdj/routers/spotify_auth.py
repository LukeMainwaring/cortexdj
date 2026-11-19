"""Spotify OAuth endpoints.

Simplified from Earworm's implementation: no user auth (single-user local app).
"""

import logging
import time
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from spotipy.oauth2 import SpotifyOAuth

from cortexdj.core.config import get_settings
from cortexdj.dependencies.db import AsyncPostgresSessionDep
from cortexdj.models.spotify_token import SpotifyToken
from cortexdj.schemas.spotify import SpotifyConnectionStatus
from cortexdj.services.spotify import SPOTIFY_SCOPES, run_spotify

logger = logging.getLogger(__name__)

config = get_settings()

spotify_auth_router = APIRouter(prefix="/spotify", tags=["spotify"])

# CSRF state cache for the OAuth flow. In-memory is fine for a single-user
# local app; move to DB or Redis if this ever runs multi-worker.
_pending_oauth_states: dict[str, float] = {}

_MAX_PENDING_STATES = 10
_STATE_TTL_SECONDS = 600  # 10 minutes


def get_oauth_manager() -> SpotifyOAuth:
    if not config.SPOTIFY_CLIENT_ID or not config.SPOTIFY_CLIENT_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Spotify is not configured. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.",
        )
    return SpotifyOAuth(
        client_id=config.SPOTIFY_CLIENT_ID,
        client_secret=config.SPOTIFY_CLIENT_SECRET,
        redirect_uri=config.SPOTIFY_REDIRECT_URI,
        scope=SPOTIFY_SCOPES,
    )


@spotify_auth_router.get("/connect")
async def connect_spotify() -> dict[str, str]:
    """Get the Spotify authorization URL to start OAuth flow."""
    oauth = get_oauth_manager()
    now = time.monotonic()
    expired = [s for s, t in _pending_oauth_states.items() if now - t > _STATE_TTL_SECONDS]
    for s in expired:
        _pending_oauth_states.pop(s, None)
    if len(_pending_oauth_states) >= _MAX_PENDING_STATES:
        _pending_oauth_states.clear()

    state = str(uuid.uuid4())
    _pending_oauth_states[state] = now
    auth_url = oauth.get_authorize_url(state=state)
    logger.info("Generated Spotify auth URL")
    return {"auth_url": auth_url}


@spotify_auth_router.get("/callback")
async def spotify_callback(
    db: AsyncPostgresSessionDep,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    """Handle Spotify OAuth callback and exchange code for tokens."""
    redirect_base = f"{config.FRONTEND_URL}/settings/spotify"

    if error:
        logger.info(f"Spotify OAuth denied: {error}")
        return RedirectResponse(f"{redirect_base}?error=access_denied")

    if not code:
        return RedirectResponse(f"{redirect_base}?error=missing_code")

    if not state or state not in _pending_oauth_states:
        return RedirectResponse(f"{redirect_base}?error=invalid_state")

    _pending_oauth_states.pop(state, None)
    oauth = get_oauth_manager()

    try:
        token_info = await run_spotify(oauth.get_access_token, code, as_dict=True, check_cache=False)

        if not token_info:
            logger.error("Failed to get access token from Spotify")
            return RedirectResponse(f"{redirect_base}?error=token_exchange_failed")

        expires_at = datetime.fromtimestamp(token_info["expires_at"], tz=UTC)

        await SpotifyToken.upsert(
            db,
            access_token=token_info["access_token"],
            refresh_token=token_info["refresh_token"],
            expires_at=expires_at,
        )

        logger.info("Successfully connected Spotify")
        return RedirectResponse(f"{redirect_base}?success=true")

    except Exception:
        logger.exception("Error during Spotify OAuth callback")
        return RedirectResponse(f"{redirect_base}?error=callback_failed")


@spotify_auth_router.get("/status")
async def get_spotify_status(db: AsyncPostgresSessionDep) -> SpotifyConnectionStatus:
    """Get the current Spotify connection status."""
    connected = await SpotifyToken.is_connected(db)
    return SpotifyConnectionStatus(connected=connected)


@spotify_auth_router.post("/disconnect")
async def disconnect_spotify(db: AsyncPostgresSessionDep) -> SpotifyConnectionStatus:
    """Disconnect Spotify by clearing stored tokens."""
    await SpotifyToken.clear(db)
    logger.info("Disconnected Spotify")
    return SpotifyConnectionStatus(connected=False)
