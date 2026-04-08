from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import spotipy
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.ml.predict import EEGModel

if TYPE_CHECKING:
    from cortexdj.schemas.thread import BrainContext


@dataclass
class AgentDeps:
    db: AsyncSession
    eeg_model: EEGModel | None = None
    spotify_client: spotipy.Spotify | None = None
    thread_id: str | None = None
    brain_context: BrainContext | None = None
