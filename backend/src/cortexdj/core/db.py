import contextlib
import functools
import logging
from collections.abc import AsyncGenerator
from typing import Annotated

import psycopg
from fastapi import Depends
from sqlalchemy import URL
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from cortexdj.core.config import get_settings

logger = logging.getLogger(__name__)

# Read once at import on purpose: the engine and its connection URL are built at import.
# get_postgres_url and async_pg_connection read this same import-time config when they run,
# so the Settings seam reaches none of them: a test points them at a database through the
# POSTGRES_* environment, set before get_settings() is first called and before this module is
# imported (tests/conftest.py). Do not reconfigure POSTGRES_* through the Settings seam, and do
# not copy this read into code a test reconfigures; the seam reaches only code that calls
# get_settings() when it runs.
config = get_settings()


def get_postgres_url(prefix: str) -> str:
    # URL.create percent-encodes the password, so one holding "@", "/", or "%" still parses.
    url = URL.create(
        prefix,
        username=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        database=config.POSTGRES_DB,
    )
    return url.render_as_string(hide_password=False)


get_async_postgres_url = functools.partial(get_postgres_url, "postgresql+asyncpg")


# Guard against stale pooled connections dropped after an idle period.
async_engine = create_async_engine(
    get_async_postgres_url(),
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=1800,
)

AsyncSessionMaker = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)


async def _get_async_sqlalchemy_session_dependency() -> AsyncGenerator[AsyncSession]:
    async with AsyncSessionMaker() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()


@contextlib.asynccontextmanager
async def get_async_sqlalchemy_session() -> AsyncGenerator[AsyncSession]:
    """A session for work outside a request (thread titling runs after the response); commits on a clean exit."""
    async for session in _get_async_sqlalchemy_session_dependency():
        yield session


AsyncPostgresSessionDep = Annotated[AsyncSession, Depends(_get_async_sqlalchemy_session_dependency)]


async def async_pg_connection() -> AsyncGenerator[psycopg.AsyncConnection]:
    # Keyword arguments, not a libpq string, so a password with spaces or quotes connects.
    async with await psycopg.AsyncConnection.connect(
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        dbname=config.POSTGRES_DB,
    ) as aconn:
        try:
            yield aconn
        finally:
            await aconn.close()


AsyncPostgresConnectionDep = Annotated[psycopg.AsyncConnection, Depends(async_pg_connection)]
