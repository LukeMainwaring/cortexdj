"""API composition root.

Aggregates each feature's router into the single ``api_router`` that ``app.py``
mounts. This is a composition-root module: it is allowed to deep-import feature
routers (the dependency-direction rule otherwise forbids cross-feature imports).
Add a new feature by importing its router here and including it below.

It also wires the seams in ``shared/`` to the slices that implement them, and
registers feature startup hooks, at import.
"""

from fastapi import APIRouter

from cortexdj.core.health import health_router
from cortexdj.core.startup import register_startup_hook
from cortexdj.features.chat.router import thread_router
from cortexdj.features.chat.stream import agent_router
from cortexdj.features.sessions import service as session_service
from cortexdj.features.sessions.eeg_model import load_eeg_model_at_startup
from cortexdj.features.sessions.router import brain_context_router, sessions_router
from cortexdj.features.spotify.router import spotify_auth_router
from cortexdj.shared.session_lookup import register_session_lookup
from cortexdj.shared.track_access import TrackAccess, register_track_access

register_track_access(
    TrackAccess(
        find_tracks_by_mood=session_service.find_tracks_by_mood,
        get_tracks=session_service.get_tracks,
        save_track_spotify_id=session_service.save_track_spotify_id,
    )
)
register_session_lookup(session_service.get_session_participant_id)

register_startup_hook(load_eeg_model_at_startup)

api_router = APIRouter()

# core
api_router.include_router(health_router)
api_router.include_router(agent_router)
api_router.include_router(thread_router)

# features
api_router.include_router(sessions_router)
api_router.include_router(brain_context_router)
api_router.include_router(spotify_auth_router)
