"""Test data builders, shared by every tier.

Hand-written (no factory-boy/polyfactory): readable, mypy-strict clean, and
copy-extendable per the repo's copy-a-slice convention. ``make_*`` builds an
unpersisted value for any tier; ``create_*`` persists through a session for the
integration tier. Builders take ``**overrides`` (or keyword arguments with defaults, where
the shape is typed) so a test states only the fields it is about. Add a builder here
rather than a private helper in a test file once two test files need the same shape.

Builders for cuttable features live inside ``# sf:`` marker blocks, imports included,
so they vanish with their feature on prune. Do not hoist those imports to the top:
keep this file importable in the backend-only variant.
"""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.features.sessions.model import EegSegment, Session, SessionTrack, ThreadBrainContext, Track


def make_session(**overrides: object) -> Session:
    """Unpersisted EEG Session: one minute of participant P01 from DEAP."""
    data: dict[str, object] = {
        "id": str(uuid4()),
        "participant_id": "P01",
        "dataset_source": "deap",
        "recorded_at": datetime(2024, 1, 1, tzinfo=UTC),
        "duration_seconds": 60.0,
    }
    data.update(overrides)
    return Session(**data)


async def create_session(db: AsyncSession, **overrides: object) -> Session:
    """Persist a Session via ``db``; it has no segments or tracks until a test adds them."""
    session = make_session(**overrides)
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


def make_track(**overrides: object) -> Track:
    """Unpersisted stimulus Track with no Spotify id yet."""
    data: dict[str, object] = {"id": str(uuid4()), "title": "Clair de Lune", "artist": "Debussy"}
    data.update(overrides)
    return Track(**data)


async def create_track(db: AsyncSession, **overrides: object) -> Track:
    """Persist a Track via ``db``."""
    track = make_track(**overrides)
    db.add(track)
    await db.flush()
    await db.refresh(track)
    return track


async def create_session_track(db: AsyncSession, session_id: str, track_id: str, **overrides: object) -> SessionTrack:
    """Persist the listener's response to an existing track within an existing session; ``relaxed`` by default."""
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
    session_track = SessionTrack(**data)
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


from typing import Any
from unittest.mock import AsyncMock

from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_evals.evaluators import EvaluatorContext
from pydantic_evals.otel import SpanTree, SpanTreeRecordingError

from cortexdj.agents.deps import AgentDeps
from cortexdj.features.chat.model import Thread


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


def _mock_session() -> AsyncSession:
    """A spec'd mock session whose lookups find no row."""
    db = AsyncMock(spec=AsyncSession)
    db.get.return_value = None
    return db


def make_deps(db: AsyncSession | None = None) -> AgentDeps:
    """Agent deps for a run; ``db`` defaults to a mock for a run that must not reach a database.

    The mock's lookups find no row, because the assistant reads the thread's brain context
    before every model request, so an assistant run in the unit tier sees none. The
    integration tier passes its real ``db_session``.
    """
    return AgentDeps(db=db if db is not None else _mock_session())


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
