"""EEG↔CLAP contrastive retrieval — session encoding and nearest-track query.

At query time a session is embedded by running its raw DEAP EEG through the
trained `EegCLAPEncoder`, mean-pooling across windows, and L2-normalizing to
produce a 512-d query vector. The vector is then compared against the
`track_audio_embeddings` pgvector index via cosine distance.

Note: sessions today correspond to full DEAP participants (all 40 trials, ~2400s
of EEG). The runtime doesn't persist raw EEG — `EegSegment` rows only store
derived features — so we re-read the `.dat` file via the session's
`participant_id`. The file-level LRU cache below avoids re-parsing the ~100 MB
pickle on every request.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.core.config import get_settings
from cortexdj.core.paths import DEAP_DATA_DIR
from cortexdj.ml.contrastive import EegCLAPEncoder, encode_session
from cortexdj.ml.contrastive_dataset import trial_to_eeg_windows
from cortexdj.ml.dataset import load_deap_participant
from cortexdj.models.session import Session
from cortexdj.models.track_audio_embedding import EMBEDDING_DIM, TrackAudioEmbedding

logger = logging.getLogger(__name__)


class DeapFileMissingError(RuntimeError):
    """A session's underlying DEAP `.dat` file can't be located on disk.

    Distinct from the encoder's missing-checkpoint case: this is a server
    misconfiguration (the sessions table references a participant whose
    data isn't available), not a transient / recoverable condition.
    """


_encoder: EegCLAPEncoder | None = None
_encoder_device: torch.device | None = None
_encoder_lock = asyncio.Lock()


@dataclass(frozen=True)
class TrackHit:
    spotify_id: str
    title: str
    artist: str
    itunes_preview_url: str | None
    similarity: float  # cosine similarity in [-1, 1], higher is closer


def _get_inference_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_encoder_sync() -> tuple[EegCLAPEncoder, torch.device]:
    checkpoint_path = Path(get_settings().CONTRASTIVE_CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        msg = (
            f"Contrastive checkpoint not found at {checkpoint_path}. "
            "Run `uv run --directory backend python -m cortexdj.ml.contrastive_train` first."
        )
        raise FileNotFoundError(msg)

    device = _get_inference_device()
    # weights_only=False is required because our checkpoint is a dict containing
    # non-tensor metadata (schema_version, config, git_commit, train/val/test
    # subject lists). The checkpoint is a trusted local artifact so the
    # deserialization risk is acceptable.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = EegCLAPEncoder().to(device)
    encoder.load_state_dict(checkpoint["state_dict"])
    encoder.eval()
    logger.info(
        "Loaded contrastive encoder from %s (schema_version=%s, git_commit=%s) on %s",
        checkpoint_path.name,
        checkpoint.get("schema_version"),
        checkpoint.get("git_commit", "unknown"),
        device,
    )
    return encoder, device


async def get_encoder() -> tuple[EegCLAPEncoder, torch.device]:
    # `torch.load` on a several-hundred-MB checkpoint releases the GIL and
    # can take multiple seconds, so without an `asyncio.Lock` + `to_thread`
    # two near-simultaneous first-hit requests would both race through the
    # `if _encoder is not None` check, both block the event loop while
    # loading, and both allocate model weights — wasting memory and stalling
    # every other request for the duration of the load.
    global _encoder, _encoder_device
    if _encoder is not None and _encoder_device is not None:
        return _encoder, _encoder_device

    async with _encoder_lock:
        if _encoder is not None and _encoder_device is not None:
            return _encoder, _encoder_device
        encoder, device = await asyncio.to_thread(_load_encoder_sync)
        _encoder = encoder
        _encoder_device = device
        return _encoder, _encoder_device


def _participant_dat_path(participant_id_str: str) -> Path:
    # Sessions are seeded with `participant_id=f"P{n:02d}"` (see seed_sessions.py),
    # which maps back to `data/deap/s{n:02d}.dat`.
    if not participant_id_str.startswith("P"):
        msg = f"unexpected participant_id format: {participant_id_str!r} (expected 'P##')"
        raise ValueError(msg)
    n = int(participant_id_str[1:])
    return DEAP_DATA_DIR / f"s{n:02d}.dat"


@functools.lru_cache(maxsize=4)
def _load_session_windows(participant_id_str: str) -> np.ndarray:
    """Return all 4s EEG windows (200Hz, 32ch) for the session as one array.

    Cached per participant because the DEAP `.dat` pickle is ~100 MB and
    re-parsing it on every request dominates retrieval latency. `maxsize=4`
    caps worst-case cache memory at ~4 × the per-session window array
    (~12 MB each), which is trivial compared to the model weights.
    """
    path = _participant_dat_path(participant_id_str)
    if not path.exists():
        msg = f"DEAP file for participant {participant_id_str} not found at {path}"
        raise DeapFileMissingError(msg)
    data, _labels = load_deap_participant(path)  # (n_trials, 32, 7680) at 128Hz
    per_trial_windows = [trial_to_eeg_windows(data[trial_idx]) for trial_idx in range(data.shape[0])]
    return np.concatenate(per_trial_windows, axis=0) if per_trial_windows else np.zeros((0, 32, 800), dtype=np.float32)


async def encode_session_to_clap_space(db: AsyncSession, session_id: str) -> np.ndarray:
    """Produce a 512-d unit query vector for a session."""
    session = await Session.get(db, session_id)
    if session is None:
        msg = f"session {session_id} not found"
        raise LookupError(msg)

    encoder, device = await get_encoder()
    # lru_cache is sync; offload to a thread so the pickle parse doesn't
    # block the event loop. Subsequent cache hits return in microseconds.
    windows = await asyncio.to_thread(_load_session_windows, session.participant_id)
    if windows.shape[0] == 0:
        msg = f"no EEG windows for session {session_id}"
        raise LookupError(msg)

    query = encode_session(encoder, windows, device)
    if query.shape != (EMBEDDING_DIM,):
        msg = f"expected query shape ({EMBEDDING_DIM},), got {query.shape}"
        raise ValueError(msg)
    return query


async def retrieve_similar_tracks(db: AsyncSession, session_id: str, *, k: int = 10) -> list[TrackHit]:
    """Top-k Spotify tracks whose CLAP audio embedding is closest to this session.

    `k` is clamped to [1, 100]. An empty list is returned if the retrieval
    index has no rows — the caller is responsible for explaining that state
    to the user.
    """
    k = max(1, min(100, k))

    # Validate the session exists first so a bogus id raises LookupError
    # (→ 404) even when the retrieval index happens to be empty. Otherwise
    # the empty-index short-circuit would mask the missing-session case.
    session = await Session.get(db, session_id)
    if session is None:
        msg = f"session {session_id} not found"
        raise LookupError(msg)

    index_count = await TrackAudioEmbedding.count(db)
    if index_count == 0:
        logger.info("retrieval skipped: track_audio_embeddings is empty")
        return []

    query = await encode_session_to_clap_space(db, session_id)
    rows_with_distance = await TrackAudioEmbedding.get_top_k_similar(db, query, k=k)

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
            "similarity": round(h.similarity, 4),
        }
        for h in hits
    ]
