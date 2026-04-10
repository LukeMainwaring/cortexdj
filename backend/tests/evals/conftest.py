"""Fixtures for brain_agent evals.

Provides ``make_fake_deps`` — a constructor for ``AgentDeps`` instances
that don't hit the real database, Spotify, or EEG model. Used by both
the deterministic ``prepare_tools`` tests (TestModel-backed) and the
real-model ``@pytest.mark.eval`` tests.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import spotipy
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.agents.deps import AgentDeps
from cortexdj.ml.predict import EEGModel


def make_fake_deps(
    *,
    spotify_client: spotipy.Spotify | None = None,
    eeg_model: EEGModel | None = None,
    thread_id: str = "test-thread",
    brain_context: Any | None = None,
) -> AgentDeps:
    fake_db = MagicMock(spec=AsyncSession)
    return AgentDeps(
        db=fake_db,
        eeg_model=eeg_model,
        spotify_client=spotify_client,
        thread_id=thread_id,
        brain_context=brain_context,
    )


def fake_spotify_client() -> spotipy.Spotify:
    return MagicMock(spec=spotipy.Spotify)


def fake_eeg_model() -> EEGModel:
    return MagicMock(spec=EEGModel)
