"""DB health endpoint runs a real SELECT 1 against Postgres."""

from httpx import AsyncClient


async def test_db_health_check_ok(client: AsyncClient) -> None:
    resp = await client.get("/api/health/db")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
