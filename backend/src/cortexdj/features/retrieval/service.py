"""EEG↔CLAP contrastive retrieval — session encoding and nearest-track query.

At query time a session is embedded by running its raw DEAP EEG through the
trained `EegCLAPEncoder`, mean-pooling across windows, and L2-normalizing to
produce a 512-d query vector. The vector is then compared against the
`track_audio_embeddings` pgvector index via cosine distance.

Note: sessions today correspond to full DEAP participants (all 40 trials, ~2400s
of EEG). The runtime doesn't persist raw EEG — `EegSegment` rows only store
derived features — so we re-read the `.dat` file via the session's
`participant_id`, read through ``shared/session_lookup.py``. The file-level LRU
cache below avoids re-parsing the ~100 MB pickle on every request.

The encoder arrives as an argument: the route and the tool read it from
``encoder.get_contrastive_encoder``, so this module never touches the slot.
"""

import asyncio
import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.core.paths import DEAP_DATA_DIR
from cortexdj.features.retrieval.audio_catalog import cache_key as audio_cache_key_for
from cortexdj.features.retrieval.encoder import ContrastiveEncoder
from cortexdj.features.retrieval.model import EMBEDDING_DIM, TrackAudioEmbedding
from cortexdj.ml.contrastive import encode_session
from cortexdj.ml.contrastive_dataset import trial_to_eeg_windows
from cortexdj.ml.dataset import load_deap_participant
from cortexdj.shared.session_lookup import get_session_participant_id

logger = logging.getLogger(__name__)


class DeapFileMissingError(RuntimeError):
    """A session's underlying DEAP `.dat` file can't be located on disk.

    Distinct from the encoder's missing-checkpoint case: this is a server
    misconfiguration (the sessions table references a participant whose
    data isn't available), not a transient / recoverable condition.
    """


class SessionNotFoundError(LookupError):
    """No session has this id. A `LookupError` so the route's 404 covers it; the tool
    tells it apart from a session with no EEG windows, which the model cannot fix by retrying."""


class ContrastiveEncoderMissingError(RuntimeError):
    """No contrastive checkpoint was loaded at startup, so a session cannot be embedded."""


@dataclass(frozen=True)
class TrackHit:
    spotify_id: str
    title: str
    artist: str
    itunes_preview_url: str | None
    audio_cache_key: str | None
    similarity: float  # cosine similarity in [-1, 1], higher is closer


def _participant_dat_path(participant_id_str: str, deap_dir: Path) -> Path:
    # Sessions are seeded with `participant_id=f"P{n:02d}"` (see seed_sessions.py),
    # which maps back to `data/deap/s{n:02d}.dat`.
    if not participant_id_str.startswith("P"):
        msg = f"unexpected participant_id format: {participant_id_str!r} (expected 'P##')"
        raise ValueError(msg)
    n = int(participant_id_str[1:])
    return deap_dir / f"s{n:02d}.dat"


@functools.lru_cache(maxsize=4)
def _load_session_windows(participant_id_str: str, deap_dir: Path) -> np.ndarray:
    """Return all 4s EEG windows (200Hz, 32ch) for the session as one array.

    Cached per participant because the DEAP `.dat` pickle is ~100 MB and
    re-parsing it on every request dominates retrieval latency. `maxsize=4`
    caps worst-case cache memory at ~4 × the per-session window array
    (~12 MB each), which is trivial compared to the model weights.
    """
    path = _participant_dat_path(participant_id_str, deap_dir)
    if not path.exists():
        msg = f"DEAP file for participant {participant_id_str} not found at {path}"
        raise DeapFileMissingError(msg)
    data, _labels = load_deap_participant(path)  # (n_trials, 32, 7680) at 128Hz
    per_trial_windows = [trial_to_eeg_windows(data[trial_idx]) for trial_idx in range(data.shape[0])]
    return np.concatenate(per_trial_windows, axis=0) if per_trial_windows else np.zeros((0, 32, 800), dtype=np.float32)


async def encode_session_to_clap_space(
    db: AsyncSession,
    session_id: str,
    encoder: ContrastiveEncoder | None,
    *,
    deap_dir: Path = DEAP_DATA_DIR,
) -> np.ndarray:
    """Produce a 512-d unit query vector for a session."""
    participant_id = await get_session_participant_id(db, session_id)
    if participant_id is None:
        msg = f"session {session_id} not found"
        raise SessionNotFoundError(msg)

    if encoder is None:
        msg = (
            "The contrastive encoder checkpoint is missing, so sessions cannot be embedded. "
            "Run `./backend/scripts/download-checkpoints.sh` to fetch it, then restart the backend."
        )
        raise ContrastiveEncoderMissingError(msg)

    # lru_cache is sync; offload to a thread so the pickle parse doesn't
    # block the event loop. Subsequent cache hits return in microseconds.
    windows = await asyncio.to_thread(_load_session_windows, participant_id, deap_dir)
    if windows.shape[0] == 0:
        msg = f"no EEG windows for session {session_id}"
        raise LookupError(msg)

    query = encode_session(encoder.model, windows, encoder.device)
    if query.shape != (EMBEDDING_DIM,):
        msg = f"expected query shape ({EMBEDDING_DIM},), got {query.shape}"
        raise ValueError(msg)
    return query


async def retrieve_similar_tracks(
    db: AsyncSession,
    session_id: str,
    encoder: ContrastiveEncoder | None,
    *,
    k: int = 10,
    deap_dir: Path = DEAP_DATA_DIR,
) -> list[TrackHit]:
    """Top-k Spotify tracks whose CLAP audio embedding is closest to this session.

    `k` is clamped to [1, 100]. An empty list is returned if the retrieval
    index has no rows — the caller is responsible for explaining that state
    to the user.
    """
    k = max(1, min(100, k))

    # Validate the session exists first so a bogus id raises SessionNotFoundError
    # (→ 404) even when the retrieval index happens to be empty. Otherwise
    # the empty-index short-circuit would mask the missing-session case.
    if await get_session_participant_id(db, session_id) is None:
        msg = f"session {session_id} not found"
        raise SessionNotFoundError(msg)

    index_count = await TrackAudioEmbedding.count(db)
    if index_count == 0:
        logger.info("retrieval skipped: track_audio_embeddings is empty")
        return []

    query = await encode_session_to_clap_space(db, session_id, encoder, deap_dir=deap_dir)
    rows_with_distance = await TrackAudioEmbedding.search_similar(db, query, k=k)

    hits: list[TrackHit] = []
    for row, distance in rows_with_distance:
        # pgvector's cosine distance is in [0, 2]; convert to similarity in [-1, 1].
        similarity = 1.0 - distance
        hits.append(
            TrackHit(
                spotify_id=row.spotify_id,
                title=row.title,
                artist=row.artist,
                itunes_preview_url=row.itunes_preview_url,
                audio_cache_key=(audio_cache_key_for(row.artist, row.title) if row.itunes_preview_url else None),
                similarity=similarity,
            )
        )
    return hits


def serialize_hits(hits: list[TrackHit]) -> list[dict[str, Any]]:
    return [
        {
            "spotify_id": h.spotify_id,
            "title": h.title,
            "artist": h.artist,
            "itunes_preview_url": h.itunes_preview_url,
            "audio_cache_key": h.audio_cache_key,
            "similarity": round(h.similarity, 4),
        }
        for h in hits
    ]
