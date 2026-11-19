"""Logfire observability (opt-in), configured once at application startup.

``logfire.configure()`` must run before any ``instrument_*`` call, and exactly
once per process — which is why this lives here rather than at the top of
``agents/brain_agent.py``. Importing an agent module is not "starting the
application": scripts, tests, and Alembic all import it, and whichever import
happened to land first silently owned the SDK's configuration.

``send_to_logfire="if-token-present"`` keeps local development and CI working
with no ``LOGFIRE_TOKEN`` set: spans are built but never exported.

Nothing here opts into payload capture. ``instrument_fastapi`` leaves
``capture_headers=False`` (its default) so Authorization headers and cookies stay
out of traces, and ``logfire.instrument_httpx`` is deliberately NOT called — it
would record the exact provider request bodies, prompts and API keys included.
Turn it on by hand for a debugging session, never by default.
"""

import logfire
from fastapi import FastAPI

from cortexdj.dependencies.db import async_engine

_configured = False


def configure_observability() -> None:
    """Configure Logfire and attach the process-wide instrumentors. Safe to call repeatedly."""
    # `python -m cortexdj.app` imports the app module twice (as __main__ and again
    # via uvicorn's import string), and the OTel instrumentors warn when attached
    # twice; reload-driven re-imports and test collection do the same.
    global _configured
    if _configured:
        return
    _configured = True
    logfire.configure(service_name="cortexdj", send_to_logfire="if-token-present")
    # The AsyncEngine is accepted directly; the instrumentor reaches its sync engine.
    logfire.instrument_sqlalchemy(engine=async_engine)
    logfire.instrument_pydantic_ai()


def instrument_app(app: FastAPI) -> None:
    """Attach the FastAPI instrumentor to one app instance.

    Separate from ``configure_observability`` because it is per-app, not per-process:
    the app object does not exist yet when the module-level configuration runs.
    """
    logfire.instrument_fastapi(app)
