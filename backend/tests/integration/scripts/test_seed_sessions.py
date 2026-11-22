"""``seed-sessions`` against the test database: a made-up participant, scored by a checkpoint trained on it here."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.features.sessions.model import EegSegment, Session, ThreadBrainContext, Track
from cortexdj.ml.band_features import band_powers, differential_entropy, windows_of
from cortexdj.ml.checkpoint import ParticipantModels, load_checkpoint
from cortexdj.ml.dataset import load_deap_participant
from cortexdj.scripts.seed_sessions import ParticipantNotTrained, seed_sessions
from cortexdj.training.command import train_classifier
from tests.factories import create_brain_context, create_thread, flatten_channel, write_deap_participant

N_TRIALS = 20


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data_dir = tmp_path_factory.mktemp("deap")
    write_deap_participant(data_dir, participant_id=1, n_trials=N_TRIALS)
    return data_dir


@pytest.fixture(scope="module")
def checkpoint_path(data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("checkpoints") / "quadrant_classifier.pt"
    train_classifier([1], data_dir, path)
    return path


def _held_out_scores(data_dir: Path, trained: ParticipantModels) -> np.ndarray:
    """``(trials, windows, 2)`` arousal and valence scores, each trial scored by the fold model that never saw it."""
    eeg, _ = load_deap_participant(data_dir / "s01.dat")
    inputs = torch.as_tensor(differential_entropy(windows_of(eeg)), dtype=torch.float32)
    with torch.no_grad():
        return np.stack(
            [
                torch.sigmoid(trained.models[fold](inputs[trial])).numpy()
                for trial, fold in enumerate(trained.fold_of_trial)
            ]
        )


async def _segments(db: AsyncSession, session_id: str) -> list[EegSegment]:
    rows = await db.scalars(
        select(EegSegment).where(EegSegment.session_id == session_id).order_by(EegSegment.segment_index)
    )
    return list(rows)


async def test_each_segment_is_scored_by_the_model_that_held_its_trial_out(
    db_session: AsyncSession, data_dir: Path, checkpoint_path: Path
) -> None:
    expected = _held_out_scores(data_dir, load_checkpoint(checkpoint_path).participants[1]).reshape(-1, 2)

    session_ids = await seed_sessions(db_session, [1], data_dir=data_dir, checkpoint_path=checkpoint_path)

    segments = await _segments(db_session, session_ids[1])
    assert len(segments) == N_TRIALS * 15
    np.testing.assert_allclose([s.arousal_score for s in segments], expected[:, 0], rtol=1e-5)
    np.testing.assert_allclose([s.valence_score for s in segments], expected[:, 1], rtol=1e-5)
    for segment in segments:
        high = (segment.arousal_score >= 0.5, segment.valence_score >= 0.5)
        assert (
            segment.dominant_state
            == {
                (True, True): "excited",
                (False, True): "relaxed",
                (True, False): "stressed",
                (False, False): "calm",
            }[high]
        )


async def test_a_segments_band_powers_are_the_models_four_bands(
    db_session: AsyncSession, data_dir: Path, checkpoint_path: Path
) -> None:
    eeg, _ = load_deap_participant(data_dir / "s01.dat")
    trial, window = 3, 7

    session_ids = await seed_sessions(db_session, [1], data_dir=data_dir, checkpoint_path=checkpoint_path)

    segment = (await _segments(db_session, session_ids[1]))[trial * 15 + window]
    assert set(segment.band_powers) == {"theta", "alpha", "beta", "gamma"}
    assert segment.band_powers == pytest.approx(band_powers(windows_of(eeg)[trial, window]))


async def test_seeding_twice_replaces_the_session_and_keeps_the_tracks(
    db_session: AsyncSession, data_dir: Path, checkpoint_path: Path
) -> None:
    first = await seed_sessions(db_session, [1], data_dir=data_dir, checkpoint_path=checkpoint_path)
    tracks = set(await db_session.scalars(select(Track.id)))

    second = await seed_sessions(db_session, [1], data_dir=data_dir, checkpoint_path=checkpoint_path)

    sessions = list(await db_session.scalars(select(Session.id).where(Session.participant_id == "P01")))
    assert sessions == [second[1]]
    assert second[1] != first[1]
    assert await _segments(db_session, first[1]) == []
    assert set(await db_session.scalars(select(Track.id))) == tracks


async def test_a_brain_context_on_the_replaced_session_loses_only_its_session_id(
    db_session: AsyncSession, data_dir: Path, checkpoint_path: Path
) -> None:
    first = await seed_sessions(db_session, [1], data_dir=data_dir, checkpoint_path=checkpoint_path)
    await create_thread(db_session, "t-seed")
    await create_brain_context(
        db_session, "t-seed", latest_session_id=first[1], dominant_mood="calm", avg_arousal=0.3, avg_valence=0.4
    )

    await seed_sessions(db_session, [1], data_dir=data_dir, checkpoint_path=checkpoint_path)

    context = await db_session.scalar(
        select(ThreadBrainContext)
        .where(ThreadBrainContext.thread_id == "t-seed")
        .execution_options(populate_existing=True)
    )
    assert context is not None
    assert (context.latest_session_id, context.dominant_mood, context.avg_arousal, context.avg_valence) == (
        None,
        "calm",
        0.3,
        0.4,
    )


async def test_a_flat_eeg_channel_fails_seeding_rather_than_storing_nan_scores(
    db_session: AsyncSession, data_dir: Path, checkpoint_path: Path, tmp_path: Path
) -> None:
    flat_dir = tmp_path / "deap"
    shutil.copytree(data_dir, flat_dir)
    flatten_channel(flat_dir / "s01.dat", trial=4, channel=9)

    with pytest.raises(ValueError, match="non-finite"):
        await seed_sessions(db_session, [1], data_dir=flat_dir, checkpoint_path=checkpoint_path)


async def test_without_a_checkpoint_seeding_says_to_train_and_writes_nothing(
    db_session: AsyncSession, data_dir: Path, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="train-classifier"):
        await seed_sessions(db_session, [1], data_dir=data_dir, checkpoint_path=tmp_path / "missing.pt")

    assert list(await db_session.scalars(select(Session.id))) == []


async def test_a_participant_the_checkpoint_lacks_says_to_train_and_writes_nothing(
    db_session: AsyncSession, data_dir: Path, checkpoint_path: Path
) -> None:
    with pytest.raises(ParticipantNotTrained, match="train-classifier"):
        await seed_sessions(db_session, [1, 2], data_dir=data_dir, checkpoint_path=checkpoint_path)

    assert list(await db_session.scalars(select(Session.id))) == []
