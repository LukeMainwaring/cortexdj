"""Integration-tier fixtures: real Postgres + HTTP, with per-test rollback.

This is the composition root for DB/HTTP tests (the test-tier analog of api.py):
new integration tests take the ``client`` and ``db_session`` fixtures from here.

Schema is created once per session by running the SHIPPED Alembic migrations
(``upgrade head``) rather than ``Base.metadata.create_all``: extensions and indexes
a model can't declare live only in the migrations, and running them validates the
migrations that ship. Each test runs inside an outer transaction rolled back on
teardown, so tests never pollute the database and never see each other's writes.
The app's session dependency is overridden to yield that same transactional
session, with its commit dropped so teardown undoes everything. A route that
commits *itself* mid-request would otherwise commit the outer transaction and
pollute the database; ``join_transaction_mode="create_savepoint"`` makes such a
commit release a SAVEPOINT inside that transaction instead, so the rollback still
covers it.

Imports here stay limited to core/shared/app (never feature code) so this file
type-checks under ``mypy --strict`` in every onboarding-prune variant. A feature
whose route hands work to a session of its own (a detached task) is outside this
rollback, so it fakes that work in its own conftest.

Caveat: ASGITransport does not run the app lifespan, so core/startup.py hooks don't
fire (the one shipped hook only logs where traces go). If a load-bearing startup
hook is added, wrap the client in ``asgi-lifespan.LifespanManager``.
"""

import os
import socket
from collections.abc import AsyncGenerator

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from cortexdj.app import app
from cortexdj.core.db import _get_async_sqlalchemy_session_dependency, get_async_postgres_url

# Relative to the backend/ working directory (where pytest runs); matches the path
# the migration scripts use (`-c src/cortexdj/alembic.ini`).
ALEMBIC_INI = "src/cortexdj/alembic.ini"


def _db_reachable() -> bool:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_PORT", "5433"))
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _is_test_db() -> bool:
    # The integration tier must use an ISOLATED test database, never a dev DB: it runs Alembic
    # migrations and test_migrations.py downgrades to base. tests/conftest.py and CI both
    # point POSTGRES_DB at `cortexdj_test`; this catches a shell that overrides it.
    return "test" in os.environ.get("POSTGRES_DB", "")


@pytest.fixture(scope="session", autouse=True)
def _require_test_db() -> None:
    # Fail, never skip: asking for this directory is asking for these tests, and a skipped
    # tier reads as a green run.
    if not _db_reachable():
        pytest.fail("integration tier needs Postgres: run ./backend/scripts/test-integration.sh", pytrace=False)
    if not _is_test_db():
        pytest.fail(
            "integration tier needs an isolated test database (POSTGRES_DB must contain 'test'): "
            "run ./backend/scripts/test-integration.sh, or set POSTGRES_DB=cortexdj_test. "
            "Refusing to run migrations against a non-test database.",
            pytrace=False,
        )


@pytest.fixture(scope="session")
def _migrated(_require_test_db: None) -> None:
    # env.py forces sqlalchemy.url from POSTGRES_*, so this upgrades whichever DB
    # those point at (cortexdj_test locally and in CI).
    command.upgrade(Config(ALEMBIC_INI), "head")


@pytest.fixture
async def engine(_migrated: None) -> AsyncGenerator[AsyncEngine]:
    # Our own engine (NullPool) so disposal is clean and we never touch the
    # module-level global engine in core/db.py. Function-scoped on purpose: a
    # session-scoped async fixture would bind to one test's event loop and break
    # the others (anyio's loop scope follows the function-scoped anyio_backend).
    eng = create_async_engine(get_async_postgres_url(), poolclass=NullPool)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest.fixture
async def db_session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession]:
    conn: AsyncConnection = await engine.connect()
    outer = await conn.begin()  # rolled back at teardown -> no pollution
    session = async_sessionmaker(
        bind=conn,
        expire_on_commit=False,
        class_=AsyncSession,
        join_transaction_mode="create_savepoint",
    )()
    try:
        yield session
    finally:
        await session.close()
        await outer.rollback()
        await conn.close()


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient]:
    async def _override() -> AsyncGenerator[AsyncSession]:
        yield db_session  # do NOT commit; teardown rolls back the outer txn

    app.dependency_overrides[_get_async_sqlalchemy_session_dependency] = _override
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides.clear()
