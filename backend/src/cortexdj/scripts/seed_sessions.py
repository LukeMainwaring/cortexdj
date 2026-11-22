"""``seed-sessions``: one EEG Session per DEAP participant, every segment scored by a held-out fold model.

Run ``train-classifier`` first: seeding reads its checkpoint and refuses to run without it.
A participant's existing DEAP session is replaced, so seeding twice leaves one session each.

Usage:
    uv run --directory backend seed-sessions
    uv run --directory backend seed-sessions --participants 1 2 3
"""

import argparse
import asyncio
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.core.db import get_async_sqlalchemy_session
from cortexdj.core.paths import DEAP_DATA_DIR
from cortexdj.features.sessions.model import EegSegment, Session, SessionTrack, ThreadBrainContext, Track
from cortexdj.ml.band_features import band_powers_from, band_variance, differential_entropy_from, windows_of
from cortexdj.ml.checkpoint import CHECKPOINT_PATH, ParticipantModels, load_checkpoint
from cortexdj.ml.dataset import load_deap_participant
from cortexdj.ml.quadrant_classifier import quadrant_of
from cortexdj.ml.recipe import RECIPE

logger = logging.getLogger(__name__)

DATASET_SOURCE = "deap"

STIMULUS_TRACKS = [
    ("Weightless", "Marconi Union"),
    ("Clair de Lune", "Claude Debussy"),
    ("Electra", "Airstream"),
    ("Mellomaniac (Chill Out Mix)", "DJ Shah"),
    ("Watermark", "Enya"),
    ("Strawberry Swing", "Coldplay"),
    ("Please Don't Go", "Barcelona"),
    ("Pure Shores", "All Saints"),
    ("Someone Like You", "Adele"),
    ("Canzonetta Sull'aria", "Mozart"),
    ("We Found Love", "Rihanna"),
    ("Levels", "Avicii"),
    ("Bangarang", "Skrillex"),
    ("Scary Monsters and Nice Sprites", "Skrillex"),
    ("Strobe", "deadmau5"),
    ("One More Time", "Daft Punk"),
    ("Around the World", "Daft Punk"),
    ("Sandstorm", "Darude"),
    ("Bohemian Rhapsody", "Queen"),
    ("Stairway to Heaven", "Led Zeppelin"),
    ("Imagine", "John Lennon"),
    ("What a Wonderful World", "Louis Armstrong"),
    ("Moonlight Sonata", "Beethoven"),
    ("Nuvole Bianche", "Ludovico Einaudi"),
    ("Experience", "Ludovico Einaudi"),
    ("Time", "Hans Zimmer"),
    ("Comptine d'un autre ete", "Yann Tiersen"),
    ("River Flows in You", "Yiruma"),
    ("The Four Seasons: Spring", "Vivaldi"),
    ("Nocturne Op.9 No.2", "Chopin"),
    ("Gymnopedie No.1", "Erik Satie"),
    ("Canon in D", "Pachelbel"),
    ("Ave Maria", "Schubert"),
    ("Adagio for Strings", "Samuel Barber"),
    ("The Blue Danube", "Johann Strauss"),
    ("In the Hall of the Mountain King", "Edvard Grieg"),
    ("Ride of the Valkyries", "Wagner"),
    ("O Fortuna", "Carl Orff"),
    ("Mars", "Gustav Holst"),
    ("1812 Overture", "Tchaikovsky"),
]


class ParticipantNotTrained(LookupError):
    """A participant the checkpoint holds no models for: seeding it would need scores no model held out."""


async def seed_sessions(
    db: AsyncSession,
    participants: list[int],
    *,
    data_dir: Path = DEAP_DATA_DIR,
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> dict[int, str]:
    """Replace each participant's DEAP session with a freshly scored one; returns the new session id by participant.

    The checkpoint and every participant are checked before anything is written, so a refusal
    leaves the database as it was. Each segment's scores come from the fold model that did not
    train on its trial, so a session looks as good as the held-out number and no better: do not
    score with any one fold model, or with a model trained on all 40 trials. The caller owns the
    transaction; the replace and the brain-context clear land in it together.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    missing = [p for p in participants if p not in checkpoint.participants]
    if missing:
        raise ParticipantNotTrained(
            f"The checkpoint at {checkpoint_path} holds no models for participants {missing}; "
            f"run train-classifier --participants {' '.join(str(p) for p in participants)}"
        )

    session_ids = {}
    for participant_id in participants:
        session_ids[participant_id] = await _replace_session(
            db, participant_id, checkpoint.participants[participant_id], data_dir
        )
        logger.info("[%d] Seeded session %s", participant_id, session_ids[participant_id])
    return session_ids


async def _replace_session(db: AsyncSession, participant_id: int, trained: ParticipantModels, data_dir: Path) -> str:
    """Delete the participant's DEAP session and insert a new one with a new id.

    Segments and session tracks go with the old session by cascade; tracks stay. A thread's
    brain context keeps a plain string id with no foreign key, so its pointer to the old
    session is cleared here, keeping its mood and averages.
    """
    eeg, _ = load_deap_participant(data_dir / f"s{participant_id:02d}.dat")
    # One band-pass per window: the model's features and the displayed band powers both come from it.
    variance = band_variance(windows_of(eeg))
    scores = _held_out_scores(variance, trained)
    participant = f"P{participant_id:02d}"

    old_ids = list(
        await db.scalars(
            select(Session.id).where(Session.participant_id == participant, Session.dataset_source == DATASET_SOURCE)
        )
    )
    if old_ids:
        await db.execute(
            update(ThreadBrainContext)
            .where(ThreadBrainContext.latest_session_id.in_(old_ids))
            .values(latest_session_id=None)
        )
        await db.execute(delete(Session).where(Session.id.in_(old_ids)))

    n_trials, n_windows = scores.shape[:2]
    window_seconds = RECIPE.window_seconds
    session_id = str(uuid.uuid4())
    db.add(
        Session(
            id=session_id,
            participant_id=participant,
            dataset_source=DATASET_SOURCE,
            recorded_at=datetime.now(UTC),
            duration_seconds=float(n_trials * n_windows * window_seconds),
            metadata_extra={"n_trials": n_trials, "sampling_rate": RECIPE.sample_rate},
        )
    )
    # The models declare no relationship, so the unit of work does not order the session's
    # insert before its segments': flush it first.
    await db.flush()
    for trial in range(n_trials):
        for window in range(n_windows):
            arousal, valence = (float(score) for score in scores[trial, window])
            index = trial * n_windows + window
            db.add(
                EegSegment(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    segment_index=index,
                    start_time=float(index * window_seconds),
                    end_time=float((index + 1) * window_seconds),
                    arousal_score=arousal,
                    valence_score=valence,
                    dominant_state=quadrant_of(arousal, valence),
                    band_powers=band_powers_from(variance[trial, window]),
                )
            )
        avg_arousal, avg_valence = (float(mean) for mean in scores[trial].mean(axis=0))
        db.add(
            SessionTrack(
                id=str(uuid.uuid4()),
                session_id=session_id,
                track_id=await _stimulus_track_id(db, trial),
                track_order=trial,
                avg_arousal=avg_arousal,
                avg_valence=avg_valence,
                dominant_state=quadrant_of(avg_arousal, avg_valence),
            )
        )
    await db.flush()
    return session_id


def _held_out_scores(variance: npt.NDArray[np.float64], trained: ParticipantModels) -> npt.NDArray[np.float32]:
    """``(trials, windows, 2)`` arousal and valence scores from the windows' band variance.

    Each score is the sigmoid of the logits of the model that held its trial out.
    """
    inputs = torch.as_tensor(differential_entropy_from(variance), dtype=torch.float32)
    with torch.no_grad():
        return np.stack(
            [
                torch.sigmoid(trained.models[fold](inputs[trial])).numpy()
                for trial, fold in enumerate(trained.fold_of_trial)
            ]
        )


async def _stimulus_track_id(db: AsyncSession, trial: int) -> str:
    """The stimulus track for ``trial``, created the first time any participant needs it."""
    title, artist = STIMULUS_TRACKS[trial % len(STIMULUS_TRACKS)]
    existing = await db.scalar(select(Track).where(Track.title == title, Track.artist == artist))
    if existing is not None:
        return existing.id
    track = Track(id=str(uuid.uuid4()), title=title, artist=artist)
    db.add(track)
    await db.flush()
    return track.id


async def _seed(participants: list[int]) -> None:
    async with get_async_sqlalchemy_session() as db:
        await seed_sessions(db, participants)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Seed CortexDJ's EEG sessions from DEAP, scored by train-classifier's checkpoint"
    )
    parser.add_argument("--participants", nargs="+", type=int, default=list(range(1, 33)))
    args = parser.parse_args()

    try:
        asyncio.run(_seed(args.participants))
    # ValueError: a checkpoint from another recipe, ratings that do not line up, or a flat EEG
    # channel. Each message says what is wrong, so a traceback adds nothing.
    except (FileNotFoundError, ParticipantNotTrained, ValueError) as error:
        raise SystemExit(f"seed-sessions: {error}") from error
    logger.info("Done: seeded %d participants", len(args.participants))
