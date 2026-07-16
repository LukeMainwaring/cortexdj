import pytest
from httpx import AsyncClient

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def test_db_health_check(client: AsyncClient) -> None:
    # Exercises the raw psycopg dependency against the real test database.
    response = await client.get("/api/health/db")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
