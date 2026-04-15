"""Seed the database with EEG session data.

Loads DEAP .dat files, runs model inference on segments,
and populates sessions, eeg_segments, tracks, and session_tracks tables.

Usage:
    uv run seed-sessions
    uv run seed-sessions --participants 1 2 3
"""

import argparse
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import select

from cortexdj.core.config import get_settings
from cortexdj.dependencies.db import get_async_sqlalchemy_session
from cortexdj.ml.dataset import (
    NUM_EEG_CHANNELS,
    SEGMENT_SAMPLES,
    load_deap_participant,
    scores_to_quadrant,
)
from cortexdj.ml.predict import EEGModel, EEGPredictionResult, load_model, predict_segment
from cortexdj.ml.preprocessing import DEFAULT_SAMPLING_RATE, compute_band_powers
from cortexdj.models.eeg_segment import EegSegment
from cortexdj.models.session import Session
from cortexdj.models.session_track import SessionTrack
from cortexdj.models.track import Track

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

config = get_settings()

# Stimulus track names for seeded sessions
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


def _load_participant_data(
    participant_id: int,
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    file_path = data_dir / f"s{participant_id:02d}.dat"
    if not file_path.exists():
        logger.warning(f"  File not found: {file_path}")
        return None
    return load_deap_participant(file_path)


async def seed_participant(
    participant_id: int,
    data_dir: Path,
    model: EEGModel | None,
) -> int:
    result = _load_participant_data(participant_id, data_dir)
    if result is None:
        return 0

    data, labels = result
    n_trials = data.shape[0]

    seeded = 0
    async with get_async_sqlalchemy_session() as db:
        session_id = str(uuid.uuid4())
        session = Session(
            id=session_id,
            participant_id=f"P{participant_id:02d}",
            dataset_source="deap",
            recorded_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            duration_seconds=float(n_trials * 60),
            metadata_extra={"n_trials": n_trials, "sampling_rate": DEFAULT_SAMPLING_RATE},
        )
        db.add(session)

        for trial_idx in range(n_trials):
            trial_data = data[trial_idx, :NUM_EEG_CHANNELS, :]
            valence = float(labels[trial_idx, 0])
            arousal = float(labels[trial_idx, 1])

            track_title, track_artist = STIMULUS_TRACKS[trial_idx % len(STIMULUS_TRACKS)]
            result_row = await db.execute(select(Track).where(Track.title == track_title, Track.artist == track_artist))
            existing_track = result_row.scalar_one_or_none()

            if existing_track is None:
                track = Track(
                    id=str(uuid.uuid4()),
                    title=track_title,
                    artist=track_artist,
                )
                db.add(track)
                await db.flush()
                track_id = track.id
            else:
                track_id = existing_track.id

            n_samples = trial_data.shape[1]
            n_segments = n_samples // SEGMENT_SAMPLES
            segment_results: list[EEGPredictionResult] = []

            for seg_idx in range(n_segments):
                start = seg_idx * SEGMENT_SAMPLES
                end = start + SEGMENT_SAMPLES
                segment_data = trial_data[:, start:end]

                if model is not None:
                    prediction = predict_segment(segment_data, model)
                else:
                    band_powers = compute_band_powers(segment_data)
                    prediction = EEGPredictionResult(
                        arousal_score=arousal / 9.0,
                        valence_score=valence / 9.0,
                        arousal_class="high" if arousal >= 5 else "low",
                        valence_class="high" if valence >= 5 else "low",
                        dominant_state=scores_to_quadrant(arousal, valence),
                        band_powers=band_powers,
                    )

                segment_results.append(prediction)

                eeg_segment = EegSegment(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    segment_index=trial_idx * n_segments + seg_idx,
                    start_time=float(trial_idx * 60 + seg_idx * 4),
                    end_time=float(trial_idx * 60 + (seg_idx + 1) * 4),
                    arousal_score=prediction.arousal_score,
                    valence_score=prediction.valence_score,
                    dominant_state=prediction.dominant_state,
                    band_powers=prediction.band_powers,
                )
                db.add(eeg_segment)

            if segment_results:
                avg_a = sum(r.arousal_score for r in segment_results) / len(segment_results)
                avg_v = sum(r.valence_score for r in segment_results) / len(segment_results)
                session_track = SessionTrack(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    track_id=track_id,
                    track_order=trial_idx,
                    avg_arousal=avg_a,
                    avg_valence=avg_v,
                    dominant_state=scores_to_quadrant(avg_a * 10, avg_v * 10),
                )
                db.add(session_track)

            seeded += 1

        await db.commit()

    return seeded


async def seed_all(
    participants: list[int],
    data_dir: str | None,
    model_type: str,
) -> None:
    resolved_dir = Path(data_dir) if data_dir else Path(config.DEAP_DATA_DIR)

    if not resolved_dir.exists():
        logger.error(f"Data directory not found: {resolved_dir}")
        logger.error("See backend/data/DEAP_SETUP.md for download instructions.")
        return

    model = None
    try:
        model = load_model(model_type=model_type)
        logger.info(f"Using trained {model_type} model for classification")
    except FileNotFoundError:
        logger.info("No trained model found. Using ground truth labels for seeding.")

    total_seeded = 0
    for p in participants:
        count = await seed_participant(p, resolved_dir, model)
        if count > 0:
            logger.info(f"  [{p}] Seeded {count} trials")
            total_seeded += count

    logger.info(f"Done! Seeded {total_seeded} total trials from {len(participants)} participants")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed CortexDJ database with EEG data")
    parser.add_argument("--participants", nargs="+", type=int, default=list(range(1, 33)))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--model",
        choices=["eegnet", "cbramod"],
        default="cbramod",
        help="Model checkpoint to use for classification (default: cbramod)",
    )
    args = parser.parse_args()

    asyncio.run(seed_all(args.participants, args.data_dir, args.model))
