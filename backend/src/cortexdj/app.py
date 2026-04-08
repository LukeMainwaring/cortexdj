import argparse
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

from cortexdj.core.config import get_settings
from cortexdj.utils.logging import RequestLogContext, setup_logging

log_context_var = setup_logging()

from cortexdj.routers.main import api_router  # noqa: E402

config = get_settings()

logger = logging.getLogger(__name__)


def generate_operation_id(route: APIRoute) -> str:
    """Generate clean camelCase operationIds for OpenAPI spec."""
    parts = route.name.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting CortexDJ backend...")

    try:
        from cortexdj.ml.predict import load_model

        model_type = config.EEG_MODEL_BACKEND
        eeg_model = load_model(model_type=model_type)
        app.state.eeg_model = eeg_model
        logger.info(f"{model_type} model loaded successfully")
    except FileNotFoundError:
        logger.warning("No EEG model checkpoint found. Run `uv run train-model` first.")
        app.state.eeg_model = None

    yield
    logger.info("Shutting down CortexDJ backend...")


app = FastAPI(
    title="CortexDJ",
    openapi_url=f"{config.API_PREFIX}/openapi.json",
    docs_url=f"{config.API_PREFIX}/docs",
    generate_unique_id_function=generate_operation_id,
    lifespan=lifespan,
)


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
            pass

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
        logger.info(f"Request: {request.method} {request.url}")
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
