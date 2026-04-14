"""DEAP ↔ iTunes-CLAP pair dataset for contrastive EEG-audio training.

Each sample is `(eeg_window, audio_embedding, trial_id, subject_id)`:
  - `eeg_window`: (32, 800) float32, a 4-second EEG segment resampled to
    200Hz to match CBraMod's pretrained input shape.
  - `audio_embedding`: (512,) float32, the CLAP embedding of the iTunes 30s
    preview of the DEAP stimulus the subject was watching during that trial.
    Computed once per unique track and shared across all EEG windows from
    the same trial.
  - `trial_id`: the DEAP Experiment_id (1..40), used by the contrastive
    loss's soft-target mask to correctly handle same-track duplicates.
  - `subject_id`: the DEAP participant id, used for train/val/test splitting.

Only (subject, trial) pairs where `deap_stimuli_resolved.json` has a cached
audio embedding survive; the ~8 DEAP stimuli iTunes couldn't resolve are
silently dropped (the training code tolerates a sparse stimulus set).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from scipy.signal import resample
from torch.utils.data import Dataset

from cortexdj.core.paths import DATA_DIR, DEAP_DATA_DIR
from cortexdj.ml.contrastive import CLAP_MODEL_ID, ClapAudioEncoder, load_audio_waveform
from cortexdj.ml.dataset import (
    CBRAMOD_SAMPLING_RATE,
    SEGMENT_SAMPLES,
    _cache_dir,
    _extract_participant_id,
    load_deap_participant,
)

logger = logging.getLogger(__name__)

_CACHE_VERSION = "v1"
STIMULI_RESOLVED_PATH = DATA_DIR / "deap_stimuli_resolved.json"

CBRAMOD_SEGMENT_SAMPLES = CBRAMOD_SAMPLING_RATE * 4  # 800 samples at 200Hz


def _audio_cache_key(resolved: list[dict[str, Any]], model_id: str) -> str:
    parts = [_CACHE_VERSION, model_id]
    for entry in sorted(resolved, key=lambda r: r["trial_id"]):
        path = Path(entry["audio_cache_path"])
        mtime = path.stat().st_mtime_ns if path.exists() else 0
        parts.append(f"{entry['trial_id']}:{path.name}:{mtime}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _audio_cache_path(key: str) -> Path:
    return _cache_dir(DEAP_DATA_DIR) / f"clap_audio_{key}.npz"


def load_resolved_stimuli(path: Path | None = None) -> list[dict[str, Any]]:
    resolved_path = path or STIMULI_RESOLVED_PATH
    if not resolved_path.exists():
        msg = (
            f"DEAP stimulus resolution output not found at {resolved_path}. "
            "Run `uv run --directory backend python -m cortexdj.scripts.fetch_deap_audio` first."
        )
        raise FileNotFoundError(msg)
    return list(json.loads(resolved_path.read_text()))


def build_audio_embedding_cache(
    resolved: list[dict[str, Any]],
    *,
    clap_encoder: ClapAudioEncoder | None = None,
    device: torch.device | None = None,
    force_rebuild: bool = False,
) -> dict[int, npt.NDArray[np.float32]]:
    """Return `{trial_id: (512,) float32}` for all resolved DEAP stimuli.

    Caches to `backend/data/deap/.cache/clap_audio_{hash}.npz`. The hash
    includes the CLAP model id and each m4a file's mtime, so touching a
    cache file or switching models invalidates cleanly.
    """
    key = _audio_cache_key(resolved, CLAP_MODEL_ID)
    cache_path = _audio_cache_path(key)

    if not force_rebuild and cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        cached_trial_ids = data["trial_ids"].astype(int)
        cached_embeddings = data["embeddings"].astype(np.float32)
        logger.info(f"Loaded {len(cached_trial_ids)} cached CLAP audio embeddings from {cache_path.name}")
        return {int(tid): cached_embeddings[i] for i, tid in enumerate(cached_trial_ids)}

    if clap_encoder is None:
        if device is None:
            from cortexdj.ml.train import _get_device

            device = _get_device()
        clap_encoder = ClapAudioEncoder(device)

    trial_ids: list[int] = []
    waveforms: list[np.ndarray] = []
    for entry in sorted(resolved, key=lambda r: r["trial_id"]):
        path = Path(entry["audio_cache_path"])
        if not path.exists():
            logger.warning(f"skipping trial {entry['trial_id']}: {path} missing")
            continue
        trial_ids.append(int(entry["trial_id"]))
        waveforms.append(load_audio_waveform(path))

    logger.info(f"Encoding {len(waveforms)} audio waveforms via CLAP...")
    embeddings = clap_encoder.embed_waveforms(waveforms)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        trial_ids=np.array(trial_ids, dtype=np.int64),
        embeddings=embeddings.astype(np.float32),
    )
    logger.info(f"Cached CLAP audio embeddings to {cache_path.name}")
    return {tid: embeddings[i] for i, tid in enumerate(trial_ids)}


class DeapClapPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int, int]]):
    """EEG window ↔ CLAP audio embedding pairs for contrastive training.

    Follows the same 4s/200Hz EEG windowing convention as DEAPRawDataset so
    the CBraMod backbone's pretrained input shape is preserved. Each DEAP
    trial contributes up to 15 non-overlapping 4-second windows from the
    60s stimulus period (baseline already stripped by load_deap_participant).
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        participants: list[int] | None = None,
        *,
        resolved_stimuli: list[dict[str, Any]] | None = None,
        audio_embeddings: dict[int, npt.NDArray[np.float32]] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else DEAP_DATA_DIR
        self.source_segment_samples = SEGMENT_SAMPLES  # 512 at 128Hz
        self.target_segment_samples = CBRAMOD_SEGMENT_SAMPLES  # 800 at 200Hz

        self.resolved_stimuli = resolved_stimuli or load_resolved_stimuli()
        self.audio_embeddings = audio_embeddings or build_audio_embedding_cache(self.resolved_stimuli)
        self.allowed_trial_ids = set(self.audio_embeddings.keys())

        if participants is None:
            files = sorted(self.data_dir.glob("*.dat"))
        else:
            files = [self.data_dir / f"s{p:02d}.dat" for p in participants]
            files = [f for f in files if f.exists()]

        if not files:
            msg = f"No DEAP .dat files found in {self.data_dir}."
            raise FileNotFoundError(msg)

        self.samples: list[tuple[npt.NDArray[np.float32], int, int]] = []
        for file_path in files:
            self._load_participant(file_path)

        logger.info(
            f"DeapClapPairDataset: {len(self.samples)} EEG windows, "
            f"{len(self.allowed_trial_ids)} unique tracks, "
            f"{len(files)} subjects"
        )

    def _load_participant(self, file_path: Path) -> None:
        subject_id = _extract_participant_id(file_path)
        data, _labels = load_deap_participant(file_path)
        n_trials = data.shape[0]

        for trial_idx in range(n_trials):
            trial_id = trial_idx + 1  # DEAP trials are 1-indexed in video_list.csv
            if trial_id not in self.allowed_trial_ids:
                continue

            trial_data = data[trial_idx]  # (32, 7680)
            n_samples = trial_data.shape[1]
            n_segments = n_samples // self.source_segment_samples

            for seg_idx in range(n_segments):
                start = seg_idx * self.source_segment_samples
                end = start + self.source_segment_samples
                segment = trial_data[:, start:end]  # (32, 512)
                resampled = resample(segment, self.target_segment_samples, axis=1).astype(np.float32)
                self.samples.append((resampled, trial_id, subject_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        segment, trial_id, subject_id = self.samples[idx]
        eeg = torch.from_numpy(segment)
        audio = torch.from_numpy(self.audio_embeddings[trial_id])
        return eeg, audio, trial_id, subject_id

    def subject_ids(self) -> list[int]:
        return sorted({s[2] for s in self.samples})

    def indices_for_subjects(self, subjects: list[int]) -> list[int]:
        allowed = set(subjects)
        return [i for i, s in enumerate(self.samples) if s[2] in allowed]
