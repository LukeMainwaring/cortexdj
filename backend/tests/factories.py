"""Test data builders, shared by every tier.

Hand-written (no factory-boy/polyfactory): readable, mypy-strict clean, and
copy-extendable per the repo's copy-a-slice convention. ``make_*`` builds an
unpersisted value for any tier; ``create_*`` persists through a session for the
integration tier. Builders take ``**overrides`` (or keyword arguments with defaults, where
the shape is typed) so a test states only the fields it is about. Add a builder here
rather than a private helper in a test file once two test files need the same shape.
"""

import csv
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import numpy as np
import numpy.typing as npt
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_evals.evaluators import EvaluatorContext
from pydantic_evals.otel import SpanTree, SpanTreeRecordingError
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.agents.deps import AgentDeps
from cortexdj.features.chat.model import Thread
from cortexdj.features.retrieval.model import EMBEDDING_DIM, TrackAudioEmbedding
from cortexdj.features.sessions.model import EegSegment, Session, SessionTrack, ThreadBrainContext, Track
from cortexdj.features.spotify.model import SpotifyToken
from cortexdj.ml.dataset import RATINGS_FILENAME


async def create_session(db: AsyncSession, **overrides: object) -> Session:
    """Persist an EEG Session via ``db``: one minute of participant P01 from DEAP, with no segments or tracks."""
    data: dict[str, object] = {
        "id": str(uuid4()),
        "participant_id": "P01",
        "dataset_source": "deap",
        "recorded_at": datetime(2024, 1, 1, tzinfo=UTC),
        "duration_seconds": 60.0,
    }
    data.update(overrides)
    session = Session(**data)
    db.add(session)
    await db.flush()
    await db.refresh(session)
    return session


def make_eeg_segment(session_id: str, **overrides: object) -> EegSegment:
    """Unpersisted first 4-second segment of ``session_id``, classified ``excited``."""
    data: dict[str, object] = {
        "id": str(uuid4()),
        "session_id": session_id,
        "segment_index": 0,
        "start_time": 0.0,
        "end_time": 4.0,
        "arousal_score": 0.7,
        "valence_score": 0.6,
        "dominant_state": "excited",
        "band_powers": {"alpha": 1.0, "beta": 0.5, "theta": 0.3, "delta": 0.2, "gamma": 0.1},
    }
    data.update(overrides)
    return EegSegment(**data)


async def create_eeg_segment(db: AsyncSession, session_id: str, **overrides: object) -> EegSegment:
    """Persist an EegSegment of an existing session via ``db``."""
    segment = make_eeg_segment(session_id, **overrides)
    db.add(segment)
    await db.flush()
    await db.refresh(segment)
    return segment


async def create_track(db: AsyncSession, **overrides: object) -> Track:
    """Persist a stimulus Track via ``db``, with no Spotify id yet."""
    data: dict[str, object] = {"id": str(uuid4()), "title": "Clair de Lune", "artist": "Debussy"}
    data.update(overrides)
    track = Track(**data)
    db.add(track)
    await db.flush()
    await db.refresh(track)
    return track


def make_session_track(session_id: str, track_id: str, **overrides: object) -> SessionTrack:
    """Unpersisted listener's response to the first track of ``session_id``; ``relaxed`` by default."""
    data: dict[str, object] = {
        "id": str(uuid4()),
        "session_id": session_id,
        "track_id": track_id,
        "track_order": 0,
        "avg_arousal": 0.3,
        "avg_valence": 0.7,
        "dominant_state": "relaxed",
    }
    data.update(overrides)
    return SessionTrack(**data)


async def create_session_track(db: AsyncSession, session_id: str, track_id: str, **overrides: object) -> SessionTrack:
    """Persist the listener's response to an existing track within an existing session; ``relaxed`` by default."""
    session_track = make_session_track(session_id, track_id, **overrides)
    db.add(session_track)
    await db.flush()
    await db.refresh(session_track)
    return session_track


async def create_brain_context(db: AsyncSession, thread_id: str, **overrides: object) -> ThreadBrainContext:
    """Persist the brain context of an existing thread; every field is None unless overridden."""
    context = ThreadBrainContext(thread_id=thread_id, **overrides)
    db.add(context)
    await db.flush()
    await db.refresh(context)
    return context


async def create_spotify_token(db: AsyncSession, **overrides: object) -> SpotifyToken:
    """Persist the user's Spotify login, valid for another hour so reading it needs no refresh."""
    data: dict[str, object] = {
        "access_token": "access-token",
        "refresh_token": "refresh-token",
        "expires_at": datetime.now(UTC) + timedelta(hours=1),
    }
    data.update(overrides)
    token = SpotifyToken(**data)
    db.add(token)
    await db.flush()
    await db.refresh(token)
    return token


async def create_track_audio_embedding(db: AsyncSession, **overrides: object) -> TrackAudioEmbedding:
    """Persist a retrieval-index row via ``db``, with no iTunes preview, its embedding the first unit vector.

    A unit vector, not zeros: cosine distance to a zero vector is undefined, which would
    poison any ordering a test asserts.
    """
    embedding = [0.0] * EMBEDDING_DIM
    embedding[0] = 1.0
    data: dict[str, object] = {
        "spotify_id": f"sp-{uuid4().hex[:16]}",
        "title": "Test Track",
        "artist": "Test Artist",
        "source": "user_library",
        "embedding": embedding,
    }
    data.update(overrides)
    row = TrackAudioEmbedding(**data)
    db.add(row)
    await db.flush()
    await db.refresh(row)
    return row


async def create_thread(db: AsyncSession, thread_id: str = "t-1", **overrides: object) -> Thread:
    """Persist a chat Thread via ``db``; it has no messages until a test appends them."""
    thread = Thread(thread_id=thread_id, **overrides)
    db.add(thread)
    await db.flush()
    await db.refresh(thread)
    return thread


def make_chat_request(thread_id: str, text: str) -> dict[str, Any]:
    """One ``/api/agent/chat`` body as the frontend sends it: the newest message only."""
    return {
        "trigger": "submit-message",
        "id": thread_id,
        "messages": [{"id": f"msg-{text}", "role": "user", "parts": [{"type": "text", "text": text}]}],
    }


def make_deps(db: AsyncSession | None = None) -> AgentDeps:
    """Agent deps for a run; ``db`` defaults to a mock for a run that must not reach a database.

    The mock's lookups find no row, because the assistant reads the thread's brain context
    before every model request, so an assistant run in the unit tier sees none. The
    integration tier passes its real ``db_session``.
    """
    if db is None:
        db = AsyncMock(spec=AsyncSession)
        db.get.return_value = None
    return AgentDeps(db=db)


def make_run_context(deps: AgentDeps, *, conversation_id: str | None = None) -> RunContext[AgentDeps]:
    """The context a tool receives, for calling a tool function directly without an agent run.

    ``conversation_id`` is the thread id the chat endpoint passes to a run; None is a run
    started outside a thread.
    """
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage(), conversation_id=conversation_id)


def make_evaluator_context[InputsT, OutputT](
    inputs: InputsT,
    output: OutputT,
    *,
    attributes: dict[str, Any] | None = None,
    span_tree: SpanTree | SpanTreeRecordingError | None = None,
) -> EvaluatorContext[InputsT, OutputT, None]:
    """The context an evaluator receives for one finished case, for running it with no dataset behind it.

    With no ``span_tree``, the context carries the error a run with no tracing records, so
    an evaluator that reads spans fails the way it would on such a run. Generic rather than
    typed to the eval tier's ``AgentTurn``: ``tests/evals/agent_run.py`` imports this module.
    """
    return EvaluatorContext(
        name="case",
        inputs=inputs,
        metadata=None,
        expected_output=None,
        output=output,
        duration=1.0,
        _span_tree=span_tree if span_tree is not None else SpanTreeRecordingError("no agent ran for this case"),
        attributes=attributes if attributes is not None else {},
        metrics={},
    )


# DEAP's preprocessed layout: 40 channels (32 EEG first), 384 baseline + 7680 stimulus samples at 128Hz.
_DEAP_CHANNELS = 40
_DEAP_SAMPLES = 8064
_RATINGS_COLUMNS = [
    "Participant_id",
    "Trial",
    "Experiment_id",
    "Start_time",
    "Valence",
    "Arousal",
    "Dominance",
    "Liking",
    "Familiarity",
]


@dataclass(frozen=True)
class FakeDeapParticipant:
    """What ``write_deap_participant`` wrote: the EEG file and the published ratings, in the file's trial order."""

    eeg_path: Path
    valence: npt.NDArray[np.float64]
    arousal: npt.NDArray[np.float64]


def write_deap_participant(
    data_dir: Path, participant_id: int = 1, *, n_trials: int = 40, seed: int = 0
) -> FakeDeapParticipant:
    """Write one made-up DEAP participant under ``data_dir``: ``sNN.dat`` and its rows of ``participant_ratings.csv``.

    The pickle's label array is altered the way the Kaggle mirror's is (valence and arousal
    are ``9 - rating`` where both ratings exceed 5) while its dominance and liking match the
    ratings, so a loader that reads valence or arousal from the pickle gets the wrong answer.
    Ratings rows are written in a shuffled presentation order, so only a join on experiment id
    lines them up. An ``n_trials`` below DEAP's 40 keeps a test's pickle small.
    """
    rng = np.random.default_rng(seed)
    ratings = rng.uniform(1, 9, (n_trials, 4)).round(2)
    ratings[0, :2] = [7.0, 8.0]  # at least one trial the mirror alters
    labels = ratings.copy()
    altered = (ratings[:, 0] > 5) & (ratings[:, 1] > 5)
    labels[altered, :2] = 9 - ratings[altered, :2]
    data = rng.standard_normal((n_trials, _DEAP_CHANNELS, _DEAP_SAMPLES)).astype(np.float32)
    eeg_path = data_dir / f"s{participant_id:02d}.dat"
    with eeg_path.open("wb") as f:
        pickle.dump({"labels": labels, "data": data}, f)
    write_participant_ratings(data_dir, participant_id, ratings, seed=seed)
    return FakeDeapParticipant(eeg_path=eeg_path, valence=ratings[:, 0], arousal=ratings[:, 1])


def write_participant_ratings(
    data_dir: Path, participant_id: int, ratings: npt.NDArray[np.float64], *, seed: int = 0
) -> None:
    """Replace ``participant_id``'s rows of ``participant_ratings.csv`` under ``data_dir`` with ``ratings``.

    ``ratings`` is ``(n_trials, 4)`` valence, arousal, dominance, liking in the EEG file's trial
    order, so row ``e`` becomes experiment id ``e + 1``. Other participants' rows are kept.
    """
    path = data_dir / RATINGS_FILENAME
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="") as f:
            rows = [row for row in csv.DictReader(f) if int(row["Participant_id"]) != participant_id]
    presentation = np.random.default_rng(seed).permutation(len(ratings))
    for trial, e in enumerate(presentation, start=1):
        valence, arousal, dominance, liking = (str(v) for v in ratings[e])
        rows.append(
            {
                "Participant_id": str(participant_id),
                "Trial": str(trial),
                "Experiment_id": str(e + 1),
                "Start_time": str(trial * 70_000),
                "Valence": valence,
                "Arousal": arousal,
                "Dominance": dominance,
                "Liking": liking,
                "Familiarity": "",
            }
        )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_RATINGS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
