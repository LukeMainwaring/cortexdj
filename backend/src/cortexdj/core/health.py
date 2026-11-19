from fastapi.routing import APIRouter

from cortexdj.core.db import AsyncPostgresConnectionDep
from cortexdj.shared.schema_base import BaseSchema


class HealthCheckResponse(BaseSchema):
    status: str


health_router = APIRouter(
    prefix="/health",
    tags=["health"],
)


@health_router.get("/db")
async def db_health_check(conn: AsyncPostgresConnectionDep) -> HealthCheckResponse:
    """Check the database connection."""
    async with conn.cursor() as cur:
        await cur.execute("SELECT 1")
    return HealthCheckResponse.model_validate({"status": "ok"})
