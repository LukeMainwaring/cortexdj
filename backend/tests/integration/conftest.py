"""Composition root for DB/HTTP integration tests — the test-tier analog of app.py.

Schema comes from the shipped Alembic migrations, not ``create_all``: the
pgvector extension and the HNSW index exist only in the migrations, so this
tier exercises the real schema. Each test runs inside an outer transaction
on a single connection that is rolled back at teardown, so tests never see
each other's writes. The ``client`` fixture overrides the app's session
dependency with that transactional session and never commits — isolation
holds only because the app's sole commit point is the overridden dependency
itself (services and model classmethods flush, never commit).

Caveats:
- ``ASGITransport`` skips lifespan, so ``app.state.eeg_model`` is never set.
  Don't integration-test ``/agent/chat``; its behavior is covered at the
  unit tier with pydantic-ai's ``TestModel``.
- ``/api/health/db`` uses the raw psycopg dependency, which is not
  overridden; it opens a real (read-only) connection to the test database.
- Guards read ``get_settings()`` rather than ``os.environ`` because settings
  may come from the repo-root ``.env``; run locally with
  ``POSTGRES_DB=cortexdj_test`` to override the ``.env`` database name.
"""

import os
import socket
from collections.abc import AsyncGenerator

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from cortexdj.core.config import get_settings
from cortexdj.dependencies.db import (
    _get_async_sqlalchemy_session_dependency,
    get_async_postgres_url,
)

_ALEMBIC_INI = "src/cortexdj/alembic.ini"  # relative to backend/, where pytest runs


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply tier markers structurally, by path.

    A forgotten per-file ``pytestmark`` would otherwise promote a DB test
    into the default (unit) tier — green in CI where Postgres exists,
    confusing failures locally.
    """
    for item in items:
        if "tests/integration" in str(item.path):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.anyio)


def _db_reachable() -> bool:
    settings = get_settings()
    try:
        with socket.create_connection((settings.POSTGRES_HOST, settings.POSTGRES_PORT), timeout=1.0):
            return True
    except OSError:
        return False


def _is_test_db() -> bool:
    # endswith, not substring: "cortexdj_latest" must not pass as a test DB —
    # this tier runs `alembic downgrade base`, which drops every table.
    return get_settings().POSTGRES_DB.endswith("_test")


@pytest.fixture(scope="session", autouse=True)
def _require_test_db() -> None:
    # In CI a misconfigured tier must fail the job, not skip silently —
    # the coverage floor alone wouldn't catch 28 skipped tests.
    in_ci = bool(os.environ.get("CI"))
    if not _db_reachable():
        message = "integration tier needs Postgres (`docker compose up -d postgres`)"
        pytest.fail(f"{message} — refusing to skip in CI") if in_ci else pytest.skip(message)
    if not _is_test_db():
        message = (
            "refusing to run against a non-test database — this tier runs alembic "
            "upgrade/downgrade; set POSTGRES_DB=cortexdj_test "
            "(see backend/scripts/create-test-db.sh)"
        )
        pytest.fail(message) if in_ci else pytest.skip(message)


@pytest.fixture(scope="session")
def _migrated(_require_test_db: None) -> None:
    cfg = Config(_ALEMBIC_INI)
    # Start pristine: wipes anything a stray seed run left in the test DB, so
    # empty-state assertions (`total == 0`) hold across runs, not just within one.
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")


@pytest.fixture
async def engine(_migrated: None) -> AsyncGenerator[AsyncEngine, None]:
    # Function-scoped with NullPool on purpose: anyio gives each test its own
    # event loop, and pooled connections bound to a previous test's loop fail.
    # Don't "optimize" this to session scope.
    engine = create_async_engine(get_async_postgres_url(), poolclass=NullPool)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def db_session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    conn = await engine.connect()
    outer = await conn.begin()  # rolled back at teardown → no cross-test pollution
    session = async_sessionmaker(bind=conn, expire_on_commit=False, class_=AsyncSession)()
    try:
        yield session
    finally:
        await session.close()
        await outer.rollback()
        await conn.close()


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    # Imported lazily so collecting the unit tier never pulls in the app.
    from cortexdj.app import app

    async def _override() -> AsyncGenerator[AsyncSession, None]:
        # Deliberately no commit: the outer transaction owns the data.
        yield db_session

    app.dependency_overrides[_get_async_sqlalchemy_session_dependency] = _override
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides.pop(_get_async_sqlalchemy_session_dependency, None)
