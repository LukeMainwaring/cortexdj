"""API composition root.

Aggregates each feature's router into the single ``api_router`` that ``app.py``
mounts. This is a composition-root module: it is allowed to deep-import feature
routers (the dependency-direction rule otherwise forbids cross-feature imports).
Add a new feature by importing its router here and including it below.
"""

from fastapi import APIRouter

from cortexdj.core.health import health_router
from cortexdj.features.chat.router import thread_router
from cortexdj.features.chat.stream import agent_router
from cortexdj.features.notes.router import notes_router

api_router = APIRouter()

# core
api_router.include_router(health_router)
api_router.include_router(agent_router)
api_router.include_router(thread_router)

# features
api_router.include_router(notes_router)
