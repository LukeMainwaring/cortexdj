"""Eval-tier fixtures: the one tier that calls real provider models.

Run the tier with ``./backend/scripts/run-evals.sh``: it loads the provider keys from ``.env``
and migrates the test database for the cases that reach a database-backed tool. The shipped
cases reach none, and the golden set needs no database at all, so ``eval_sessions`` is
requested by name rather than autouse. A case that needs it fails, not skips, when the
database is missing, for the same reason the integration tier does.

Logfire is configured here because the span-based evaluators (``MaxCallsOf``, through the
library's ``MaxToolCalls``) read the tool calls of a run from its OpenTelemetry spans. Without
a configured tracer there is no span tree and they fail every case. No token is needed for
that; when ``LOGFIRE_TOKEN`` is set the runs are also exported, under environment ``eval``.
The app's ``setup_observability`` is not reused: it takes an app object and attaches the
SQLAlchemy and FastAPI instrumentors, and the evals import no app. Its scrubbing backstop is,
so an identifier in a case's prompt or a tool's answer leaves the process as a placeholder.
"""

import os
from collections.abc import AsyncIterator, Iterator

import logfire
import pytest
from pydantic_ai import models
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from cortexdj.core import observability
from cortexdj.core.config import get_settings
from cortexdj.core.db import get_async_postgres_url

_RUN_SCRIPT_HINT = "run ./backend/scripts/run-evals.sh"


def pytest_configure(config: pytest.Config) -> None:
    # Nothing closes the judge model's HTTP client before the process exits, and what it holds
    # warns at interpreter shutdown. Under the suite's warnings-as-errors setting that would
    # fail the run after the tests had passed.
    config.addinivalue_line("filterwarnings", "ignore::ResourceWarning")


@pytest.fixture(autouse=True)
def _allow_model_requests() -> Iterator[None]:
    """Lift the suite-wide block on real model requests (tests/conftest.py) for evals."""
    with models.override_allow_model_requests(True):
        yield


@pytest.fixture(scope="session", autouse=True)
def _configure_logfire() -> None:
    token = get_settings().LOGFIRE_TOKEN
    logfire.configure(
        service_name=observability.SERVICE_NAME,
        environment="eval",
        token=token.get_secret_value() if token else None,
        send_to_logfire="if-token-present",
        console=False,  # the eval report is the output; a span per model call would bury it
    )
    logfire.instrument_pydantic_ai()


@pytest.fixture
async def eval_sessions() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory on the migrated test database, for a case that reaches a database-backed tool.

    Each case run opens its own session from this, because cases run concurrently and an
    ``AsyncSession`` cannot. The engine is per test and unpooled for the reason given in
    ``tests/integration/conftest.py``: a pooled connection would outlive the test's event loop.
    Nothing rolls a case back, since concurrent cases share no transaction: remove what they
    write in ``run-evals.sh``'s ``clean_up``, so the integration tier still finds the test DB empty.
    """
    if "test" not in os.environ.get("POSTGRES_DB", ""):
        pytest.fail(f"evals write to the database, so POSTGRES_DB must be a test one: {_RUN_SCRIPT_HINT}")
    engine = create_async_engine(get_async_postgres_url(), poolclass=NullPool)
    try:
        try:
            async with engine.connect() as connection:
                # A reachable but unmigrated database would fail mid-run, as a scored failure.
                await connection.execute(text("SELECT version_num FROM alembic_version"))
        except Exception as error:
            pytest.fail(f"evals need the migrated test database ({error}): {_RUN_SCRIPT_HINT}", pytrace=False)
        yield async_sessionmaker(bind=engine, expire_on_commit=False)
    finally:
        await engine.dispose()
