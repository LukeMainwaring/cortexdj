from fastapi import APIRouter

from cortexdj.routers.agent import agent_router
from cortexdj.routers.health import health_router
from cortexdj.routers.retrieval import retrieval_router
from cortexdj.routers.sessions import sessions_router
from cortexdj.routers.spotify_auth import spotify_auth_router
from cortexdj.routers.thread import thread_router

api_router = APIRouter()
api_router.include_router(agent_router)
api_router.include_router(health_router)
api_router.include_router(retrieval_router)
api_router.include_router(sessions_router)
api_router.include_router(spotify_auth_router)
api_router.include_router(thread_router)
