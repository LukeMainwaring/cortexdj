"""Route-level test for the similar-tracks endpoint's 404 path.

Deliberately no happy-path HTTP test: retrieval needs the contrastive
checkpoint and DEAP ``.dat`` files, neither of which exists in CI. The
503 (missing checkpoint) and 500 (missing DEAP file) mappings are covered
at the unit tier in ``tests/unit/services/test_retrieval_service.py`` and
``tests/unit/agents/test_retrieval_tool.py``.
"""

from httpx import AsyncClient


async def test_similar_tracks_for_unknown_session_returns_404(client: AsyncClient) -> None:
    response = await client.get("/api/sessions/no-such-session/similar-tracks")
    assert response.status_code == 404
