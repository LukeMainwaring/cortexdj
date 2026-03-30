from typing import Annotated

import psycopg
from fastapi import Depends
from fastapi.routing import APIRouter

from cortexdj.dependencies.db import async_pg_connection
from cortexdj.schemas.health_check import HealthCheckResponse

health_router = APIRouter(
    prefix="/health",
    tags=["health"],
)


@health_router.get("/db")
async def db_health_check(
    conn: Annotated[psycopg.AsyncConnection, Depends(async_pg_connection)],
) -> HealthCheckResponse:
    """Check the database connection."""
    async with conn.cursor() as cur:
        await cur.execute("SELECT 1")
    return HealthCheckResponse.model_validate({"status": "ok"})
