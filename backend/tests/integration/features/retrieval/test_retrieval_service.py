"""Retrieval against the real database: session lookup, the empty index, k clamping, and the ranked hits.

The session's participant id comes through the real session-lookup seam (``api.py`` registers
it when the integration conftest imports the app). The encoder is passed as the argument and
the DEAP files come from ``deap_dir=``, a directory the test writes under ``tmp_path``.
"""

from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.features.retrieval import service as retrieval_service
from cortexdj.features.retrieval.encoder import ContrastiveEncoder
from cortexdj.features.retrieval.model import EMBEDDING_DIM
from tests.factories import create_session, create_track_audio_embedding


async def _create_rows(db: AsyncSession, count: int) -> None:
    for i in range(count):
        await create_track_audio_embedding(db, spotify_id=f"sp-{i:03d}")


async def test_an_unknown_session_with_an_empty_index_is_not_found(db_session: AsyncSession) -> None:
    with pytest.raises(retrieval_service.SessionNotFoundError, match="no-such-session"):
        await retrieval_service.retrieve_similar_tracks(db_session, "no-such-session", None)


async def test_an_unknown_session_with_a_populated_index_is_not_found(
    db_session: AsyncSession, encoder: ContrastiveEncoder, deap_dir: Path
) -> None:
    await _create_rows(db_session, 1)

    with pytest.raises(retrieval_service.SessionNotFoundError, match="no-such-session"):
        await retrieval_service.retrieve_similar_tracks(db_session, "no-such-session", encoder, deap_dir=deap_dir)


async def test_an_empty_index_returns_no_hits_without_an_encoder(db_session: AsyncSession, tmp_path: Path) -> None:
    session = await create_session(db_session, participant_id="P01")

    # No encoder and no DEAP file: an empty index never reaches the embedding step.
    hits = await retrieval_service.retrieve_similar_tracks(db_session, session.id, None, deap_dir=tmp_path)

    assert hits == []


async def test_no_encoder_with_a_populated_index_says_how_to_fetch_the_checkpoint(
    db_session: AsyncSession, deap_dir: Path
) -> None:
    session = await create_session(db_session, participant_id="P01")
    await _create_rows(db_session, 1)

    with pytest.raises(retrieval_service.ContrastiveEncoderMissingError, match="download-checkpoints.sh"):
        await retrieval_service.retrieve_similar_tracks(db_session, session.id, None, deap_dir=deap_dir)


async def test_a_participant_with_no_deap_file_is_reported_with_its_path(
    db_session: AsyncSession, encoder: ContrastiveEncoder, tmp_path: Path
) -> None:
    session = await create_session(db_session, participant_id="P99")
    await _create_rows(db_session, 1)

    with pytest.raises(retrieval_service.DeapFileMissingError, match="s99.dat"):
        await retrieval_service.retrieve_similar_tracks(db_session, session.id, encoder, deap_dir=tmp_path)


async def test_k_below_one_returns_one_hit(
    db_session: AsyncSession, encoder: ContrastiveEncoder, deap_dir: Path
) -> None:
    session = await create_session(db_session, participant_id="P01")
    await _create_rows(db_session, 3)

    hits = await retrieval_service.retrieve_similar_tracks(db_session, session.id, encoder, k=0, deap_dir=deap_dir)

    assert len(hits) == 1


async def test_k_above_one_hundred_returns_one_hundred_hits(
    db_session: AsyncSession, encoder: ContrastiveEncoder, deap_dir: Path
) -> None:
    session = await create_session(db_session, participant_id="P01")
    await _create_rows(db_session, 101)

    hits = await retrieval_service.retrieve_similar_tracks(db_session, session.id, encoder, k=500, deap_dir=deap_dir)

    assert len(hits) == 100


async def test_hits_are_ranked_nearest_first_with_cosine_similarity(
    db_session: AsyncSession, encoder: ContrastiveEncoder, deap_dir: Path
) -> None:
    session = await create_session(db_session, participant_id="P01")
    # The encoder's weights are random, so the query is read back from the service and the
    # index is built around it: the query itself, a vector orthogonal to it, and its opposite.
    query = await retrieval_service.encode_session_to_clap_space(db_session, session.id, encoder, deap_dir=deap_dir)
    other = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    other[0] = 1.0
    orthogonal = other - float(np.dot(other, query)) * query
    await create_track_audio_embedding(db_session, spotify_id="sp-opposite", embedding=(-query).tolist())
    await create_track_audio_embedding(db_session, spotify_id="sp-orthogonal", embedding=orthogonal.tolist())
    await create_track_audio_embedding(db_session, spotify_id="sp-same", embedding=query.tolist())

    hits = await retrieval_service.retrieve_similar_tracks(db_session, session.id, encoder, k=3, deap_dir=deap_dir)

    assert [h.spotify_id for h in hits] == ["sp-same", "sp-orthogonal", "sp-opposite"]
    assert [h.similarity for h in hits] == pytest.approx([1.0, 0.0, -1.0], abs=1e-4)


async def test_a_hit_has_an_audio_cache_key_only_when_its_row_has_a_preview(
    db_session: AsyncSession, encoder: ContrastiveEncoder, deap_dir: Path
) -> None:
    session = await create_session(db_session, participant_id="P01")
    await create_track_audio_embedding(
        db_session,
        spotify_id="sp-preview",
        title="Teardrop",
        artist="Massive Attack",
        itunes_preview_url="https://audio.example.com/teardrop.m4a",
    )
    await create_track_audio_embedding(db_session, spotify_id="sp-no-preview", title="Angel", artist="Massive Attack")

    hits = await retrieval_service.retrieve_similar_tracks(db_session, session.id, encoder, k=2, deap_dir=deap_dir)

    keys = {h.spotify_id: h.audio_cache_key for h in hits}
    assert keys == {"sp-preview": "c534f229f38ff5bc9a75f058014e736f83beddf6", "sp-no-preview": None}
