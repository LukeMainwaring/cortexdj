"""Logfire observability: the one entrypoint that instruments FastAPI, SQLAlchemy, and Pydantic AI.

``setup_observability(app)`` runs once, right after ``app.py`` builds the FastAPI
object. ``logfire.configure`` goes first (an ``instrument_*`` call made before it
registers a hook whose spans go nowhere), then the instrumentors. Nothing is
exported unless ``LOGFIRE_TOKEN`` is set, read through ``Settings`` so ``.env`` works
for a bare ``uv run`` as well as Compose (``send_to_logfire="if-token-present"``), so
local development and tests run with no account; spans still print to the console
until ``LOGFIRE_CONSOLE=false`` silences them.

Do not move the FastAPI instrumentor into a startup hook: it works by patching
``build_middleware_stack``, which Starlette runs on the first ASGI call, *before*
the lifespan; attached from a hook it would never take effect. The startup hook
this module registers only reports at boot where traces are going.

Keep this to the standard integration. Anything a trace should be searchable by that
the instrumentors cannot know rides on the agent run as ``metadata``; do not add span
helpers here or spans for what the instrumentors already record
(docs/adr/0003-run-metadata-not-a-custom-chat-turn-span.md).
"""

import logging
from typing import Any

import logfire
from fastapi import FastAPI
from opentelemetry.trace import Span
from starlette.datastructures import URL, QueryParams

from cortexdj.core.config import get_settings
from cortexdj.core.db import async_engine
from cortexdj.core.logging import redact_query_params, redact_url
from cortexdj.core.startup import register_startup_hook

logger = logging.getLogger(__name__)

# Read once at import on purpose: Logfire is configured once, as app.py is imported. Do
# not copy this into code a test reconfigures; the Settings seam reaches only code that
# calls get_settings() when it runs.
config = get_settings()

SERVICE_NAME = "cortexdj"

# Request-span attributes that carry the raw query string (old and stable HTTP semconv).
_URL_ATTRIBUTES = ("http.url", "url.full")
_QUERY_ATTRIBUTES = ("url.query",)


def redact_request_span_url(span: Span, scope: dict[str, Any]) -> None:
    """Apply the request log's credential redaction (``core/logging.py``) to the request span.

    Logfire's scrubber treats URL attributes as safe and never pattern-matches them, so a
    credential query parameter would reach the exporter as sent. This runs as the ASGI
    instrumentor's ``server_request_hook``, after it has set the request attributes (a span
    processor's ``on_start`` is too early: the instrumentor re-sets them right after). Logfire
    derives the span's message from ``http.url`` at export, so that is redacted too.
    """
    if not span.is_recording():
        return
    attributes = getattr(span, "attributes", None) or {}
    for key in _URL_ATTRIBUTES:
        value = attributes.get(key)
        if isinstance(value, str) and "?" in value:
            span.set_attribute(key, redact_url(URL(value)))
    for key in _QUERY_ATTRIBUTES:
        value = attributes.get(key)
        if isinstance(value, str) and value:
            span.set_attribute(key, str(redact_query_params(QueryParams(value))))


_configured = False


def setup_observability(app: FastAPI) -> None:
    """Configure Logfire and attach every instrumentor; call once per app object.

    The process-wide half (configure, SQLAlchemy, Pydantic AI) is guarded because
    ``python -m cortexdj.app`` imports the app module twice (as ``__main__`` and
    via uvicorn) and the SQLAlchemy instrumentor would attach its engine listeners
    twice. The FastAPI half is per app, so each of those two app objects gets it.
    """
    global _configured
    if not _configured:
        _configured = True
        logfire.configure(
            service_name=SERVICE_NAME,
            environment=config.ENVIRONMENT,
            token=config.LOGFIRE_TOKEN.get_secret_value() if config.LOGFIRE_TOKEN else None,
            send_to_logfire="if-token-present",
        )
        logfire.instrument_sqlalchemy(engine=async_engine)
        logfire.instrument_pydantic_ai()
    logfire.instrument_fastapi(app, server_request_hook=redact_request_span_url)


async def report_observability_status(app: FastAPI) -> None:
    """Startup hook: say at boot whether traces leave the process, so a missing token is not a mystery."""
    if logfire.DEFAULT_LOGFIRE_INSTANCE.config.token:
        logger.info("Logfire: LOGFIRE_TOKEN set; exporting traces to the configured project")
    else:
        logger.info("Logfire: no LOGFIRE_TOKEN; traces stay local (console only)")


register_startup_hook(report_observability_status)
