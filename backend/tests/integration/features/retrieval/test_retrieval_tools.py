"""The retrieval agent tool against the real service and database, called the way the agent calls it.

The tool reads the encoder from the holder (``override_contrastive_encoder``) and the DEAP
files from the app's data directory, which it cannot be pointed away from; so no test here
reaches the ranked path. That path is covered at the service tier, where ``deap_dir=`` is an
argument. Do not read or write ``backend/data/`` to cover it here.
"""

import json

import pytest
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.features.retrieval import tools as retrieval_tools
from cortexdj.features.retrieval.encoder import ContrastiveEncoder, override_contrastive_encoder
from tests.factories import create_session, create_track_audio_embedding, make_deps, make_run_context


async def test_an_empty_index_returns_no_tracks_and_a_note_naming_the_seed_command(db_session: AsyncSession) -> None:
    session = await create_session(db_session, participant_id="P01")

    with override_contrastive_encoder(None):
        payload = json.loads(
            await retrieval_tools.retrieve_tracks_from_brain_state(make_run_context(make_deps(db_session)), session.id)
        )

    assert payload["session_id"] == session.id
    assert payload["k"] == 10
    assert payload["tracks"] == []
    assert "seed-track-index" in payload["note"]


async def test_k_is_echoed_unchanged_even_out_of_range(db_session: AsyncSession) -> None:
    session = await create_session(db_session, participant_id="P01")

    with override_contrastive_encoder(None):
        payload = json.loads(
            await retrieval_tools.retrieve_tracks_from_brain_state(
                make_run_context(make_deps(db_session)), session.id, k=500
            )
        )

    # The service clamps k for the query; the payload carries what the model asked for.
    assert payload["k"] == 500


async def test_an_unknown_session_asks_for_a_listed_id(db_session: AsyncSession) -> None:
    with pytest.raises(ModelRetry, match="list_sessions"):
        await retrieval_tools.retrieve_tracks_from_brain_state(
            make_run_context(make_deps(db_session)), "no-such-session"
        )


async def test_a_missing_deap_file_fails_with_the_services_message(
    db_session: AsyncSession, encoder: ContrastiveEncoder
) -> None:
    # No DEAP directory anywhere holds s99.dat, so this never depends on what is on disk.
    session = await create_session(db_session, participant_id="P99")
    await create_track_audio_embedding(db_session)

    with override_contrastive_encoder(encoder), pytest.raises(ToolFailed) as failure:
        await retrieval_tools.retrieve_tracks_from_brain_state(make_run_context(make_deps(db_session)), session.id)

    assert "DEAP file for participant P99 not found" in str(failure.value)


async def test_no_encoder_fails_with_how_to_fetch_the_checkpoint(db_session: AsyncSession) -> None:
    session = await create_session(db_session, participant_id="P01")
    await create_track_audio_embedding(db_session)

    with override_contrastive_encoder(None), pytest.raises(ToolFailed) as failure:
        await retrieval_tools.retrieve_tracks_from_brain_state(make_run_context(make_deps(db_session)), session.id)

    assert "download-checkpoints.sh" in str(failure.value)
