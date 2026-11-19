import argparse
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

from cortexdj.core.config import get_settings
from cortexdj.core.logging import RequestLogContext, redact_url, setup_logging
from cortexdj.core.observability import setup_observability
from cortexdj.core.startup import get_shutdown_hooks, get_startup_hooks

# Logging and observability are process configuration that must be in place before the
# lifespan runs, so they are set up at import here rather than as startup hooks.
log_context_var = setup_logging()

# Do not hoist this import above setup_logging(): it imports every feature, each module
# creates its logger at import, and only a logger created after setup_logging() sets the
# logger class is a ContextualLogger that carries the request context.
from cortexdj.api import api_router  # noqa: E402

# Read once at import on purpose: the app, its routes, and CORS are built from it here. Do
# not copy this into code a test reconfigures; the Settings seam reaches only code that
# calls get_settings() when it runs.
config = get_settings()

logger = logging.getLogger(__name__)


def generate_operation_id(route: APIRoute) -> str:
    """Generate clean camelCase operationIds for OpenAPI spec."""
    parts = route.name.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting CortexDJ backend...")

    # Run the registered startup hooks (core/startup.py); features register theirs at import.
    for hook in get_startup_hooks():
        await hook(app)

    yield
    logger.info("Shutting down CortexDJ backend...")
    for hook in get_shutdown_hooks():
        # One hook failing must not stop the rest: each releases something of its own.
        try:
            await hook(app)
        except Exception:
            logger.exception(f"Shutdown hook {getattr(hook, '__name__', repr(hook))} failed")


app = FastAPI(
    title="CortexDJ",
    openapi_url=f"{config.API_PREFIX}/openapi.json",
    docs_url=f"{config.API_PREFIX}/docs",
    generate_unique_id_function=generate_operation_id,
    lifespan=lifespan,
)
# After the `api` import: no module instruments at import, so configure may follow it.
setup_observability(app)


def _get_allowed_origins() -> list[str]:
    return config.ALLOWED_ORIGINS.get(config.ENVIRONMENT, config.ALLOWED_ORIGINS["development"])


app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router, prefix=config.API_PREFIX)


@app.middleware("http")
async def add_request_context(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    request_json = None
    if request.method in {"POST", "PUT", "PATCH"}:
        try:
            request_json = await request.json()
        except Exception:
            pass  # a body that isn't JSON just means there is nothing to add to the log context

    ctx = RequestLogContext(
        request_id=uuid.uuid4(),
        request=request,
        request_json=request_json,
    )
    token = log_context_var.set(ctx)
    try:
        return await call_next(request)
    finally:
        log_context_var.reset(token)


@app.middleware("http")
async def log_request(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    url_path = str(request.url.path)
    noisy_patterns = ["/api/health/", "/api/health"]
    if not any(url_path.endswith(pattern) for pattern in noisy_patterns) and request.method != "OPTIONS":
        logger.info(f"Request: {request.method} {redact_url(request.url)}")
    return await call_next(request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    uvicorn.run(
        "cortexdj.app:app",
        host="127.0.0.1",
        port=args.port,
        reload=args.reload,
    )
