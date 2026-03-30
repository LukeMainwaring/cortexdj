from __future__ import annotations

import contextlib
import functools
import logging
from typing import Annotated, AsyncGenerator

import psycopg
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from cortexdj.core.config import get_settings

logger = logging.getLogger(__name__)

config = get_settings()

_user = config.POSTGRES_USER
_password = config.POSTGRES_PASSWORD
_host = config.POSTGRES_HOST
_port = config.POSTGRES_PORT
_db = config.POSTGRES_DB

_conn_info = f"user={_user} password={_password} host={_host} port={_port} dbname={_db}"


def get_postgres_url(prefix: str) -> str:
    return f"{prefix}://{_user}:{_password}@{_host}:{_port}/{_db}"


get_async_postgres_url = functools.partial(get_postgres_url, "postgresql+asyncpg")


async_engine = create_async_engine(get_async_postgres_url(), pool_size=20, max_overflow=10)

AsyncSessionMaker = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)


async def _get_async_sqlalchemy_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionMaker() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()


class SessionMakerDep:
    @contextlib.asynccontextmanager
    async def new_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        async for session in _get_async_sqlalchemy_session_dependency():
            yield session


@contextlib.asynccontextmanager
async def get_async_sqlalchemy_session() -> AsyncGenerator[AsyncSession, None]:
    async for session in _get_async_sqlalchemy_session_dependency():
        yield session


AsyncPostgresSessionDep = Annotated[AsyncSession, Depends(_get_async_sqlalchemy_session_dependency)]
AsyncSessionMakerDep = Annotated[SessionMakerDep, Depends(SessionMakerDep)]


async def async_pg_connection() -> AsyncGenerator[psycopg.AsyncConnection, None]:
    async with await psycopg.AsyncConnection.connect(_conn_info) as aconn:
        try:
            yield aconn
        finally:
            await aconn.close()


AsyncPostgresConnectionDep = Annotated[psycopg.AsyncConnection, Depends(async_pg_connection)]
