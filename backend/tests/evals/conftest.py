"""Fixtures for brain_agent evals.

Provides ``make_fake_deps`` — a constructor for ``AgentDeps`` instances
that don't hit the real database, Spotify, or EEG model. Used by both
the deterministic ``prepare_tools`` tests (TestModel-backed) and the
real-model ``@pytest.mark.eval`` tests.
"""

from __future__ import annotations

import os

# Must be set before importing brain_agent — OpenAIResponsesModel eagerly
# constructs its provider client at module-import time. Deterministic tests
# swap in TestModel via agent.override() and never hit OpenAI; the dummy
# key exists only so the provider constructor doesn't raise.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-deterministic-no-real-calls")

from typing import Any  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from cortexdj.agents.deps import AgentDeps  # noqa: E402


def make_fake_deps(
    *,
    spotify_client: Any | None = None,
    eeg_model: Any | None = None,
    thread_id: str = "test-thread",
    brain_context: Any | None = None,
) -> AgentDeps:
    """Build AgentDeps with mock stand-ins for external services.

    The ``db`` field is a ``MagicMock`` spec'd to AsyncSession — safe as
    long as tests don't exercise tool bodies that actually query the DB.
    Tools that hit the DB are covered by the real-model eval suite, which
    runs against a seeded test database if configured, or is skipped.
    """
    from sqlalchemy.ext.asyncio import AsyncSession

    fake_db = MagicMock(spec=AsyncSession)

    return AgentDeps(
        db=fake_db,
        eeg_model=eeg_model,
        spotify_client=spotify_client,
        thread_id=thread_id,
        brain_context=brain_context,
    )
