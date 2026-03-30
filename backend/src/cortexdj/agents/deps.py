"""Shared dependencies for the CortexDJ brain assistant agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import spotipy
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.ml.model import EEGNetClassifier

if TYPE_CHECKING:
    from cortexdj.schemas.thread import BrainContext


@dataclass
class AgentDeps:
    """Dependencies injected into agent runs."""

    db: AsyncSession
    eeg_model: EEGNetClassifier | None = None
    spotify_client: spotipy.Spotify | None = None
    thread_id: str | None = None
    brain_context: BrainContext | None = None
