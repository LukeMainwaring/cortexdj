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

from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.features.notes.model import Note


def make_note(**overrides: object) -> Note:
    """Unpersisted Note with sensible defaults; override any field (incl. ``embedding``)."""
    data: dict[str, object] = {"title": "Groceries", "content": "milk, eggs, bread"}
    data.update(overrides)
    return Note(**data)


async def create_note(db: AsyncSession, **overrides: object) -> Note:
    """Persist a Note via ``db`` and return it (bypasses the embedder)."""
    note = make_note(**overrides)
    db.add(note)
    await db.flush()
    await db.refresh(note)
    return note


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


def make_run_context(deps: AgentDeps) -> RunContext[AgentDeps]:
    """The context a tool receives, for calling a tool function directly without an agent run."""
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage())


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
