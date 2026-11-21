"""Route-level tests for the similar-tracks and audio-preview endpoints, asserted through HTTP.

The route reads its DEAP files from the app's data directory, so the ranked happy path is
covered at the service tier (``test_retrieval_service.py``), where the directory is an
argument. The encoder is replaced through ``app.dependency_overrides[get_contrastive_encoder]``.
"""

from collections.abc import Iterator

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.app import app
from cortexdj.features.retrieval.encoder import ContrastiveEncoder, get_contrastive_encoder
from tests.factories import create_session, create_track_audio_embedding


@pytest.fixture
def no_encoder() -> Iterator[None]:
    app.dependency_overrides[get_contrastive_encoder] = lambda: None
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_contrastive_encoder, None)


@pytest.fixture
def fake_encoder(encoder: ContrastiveEncoder) -> Iterator[None]:
    app.dependency_overrides[get_contrastive_encoder] = lambda: encoder
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_contrastive_encoder, None)


async def test_similar_tracks_for_unknown_session_returns_404(client: AsyncClient) -> None:
    response = await client.get("/api/sessions/no-such-session/similar-tracks")
    assert response.status_code == 404


async def test_similar_tracks_with_an_empty_index_returns_no_tracks(
    client: AsyncClient, db_session: AsyncSession, no_encoder: None
) -> None:
    session = await create_session(db_session, participant_id="P01")

    response = await client.get(f"/api/sessions/{session.id}/similar-tracks", params={"k": 5})

    assert response.status_code == 200
    assert response.json() == {"session_id": session.id, "tracks": [], "k": 5}


async def test_similar_tracks_without_an_encoder_returns_503(
    client: AsyncClient, db_session: AsyncSession, no_encoder: None
) -> None:
    session = await create_session(db_session, participant_id="P01")
    await create_track_audio_embedding(db_session)

    response = await client.get(f"/api/sessions/{session.id}/similar-tracks")

    assert response.status_code == 503
    assert "download-checkpoints.sh" in response.json()["detail"]


async def test_similar_tracks_for_a_participant_without_deap_data_returns_500(
    client: AsyncClient, db_session: AsyncSession, fake_encoder: None
) -> None:
    session = await create_session(db_session, participant_id="P99")
    await create_track_audio_embedding(db_session)

    response = await client.get(f"/api/sessions/{session.id}/similar-tracks")

    assert response.status_code == 500
    assert "DEAP file for participant P99 not found" in response.json()["detail"]


@pytest.mark.parametrize("k", [0, 101])
async def test_similar_tracks_with_k_out_of_range_returns_422(
    client: AsyncClient, db_session: AsyncSession, k: int
) -> None:
    session = await create_session(db_session, participant_id="P01")

    response = await client.get(f"/api/sessions/{session.id}/similar-tracks", params={"k": k})

    assert response.status_code == 422


async def test_audio_preview_with_a_malformed_key_returns_400(client: AsyncClient) -> None:
    response = await client.get("/api/audio/preview/not-a-cache-key")

    assert response.status_code == 400


async def test_audio_preview_that_is_not_cached_returns_404(client: AsyncClient) -> None:
    # A well-formed SHA-1 key that no artist/title hashes to in practice.
    response = await client.get(f"/api/audio/preview/{'0' * 40}")

    assert response.status_code == 404
