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

from cortexdj.core.paths import AUDIO_CACHE_DIR, DATA_DIR, DEAP_DATA_DIR
from cortexdj.ml.contrastive import CLAP_MODEL_ID, ClapAudioEncoder, load_audio_waveform
from cortexdj.ml.dataset import (
    CBRAMOD_SAMPLING_RATE,
    SEGMENT_SAMPLES,
    _cache_dir,
    _extract_participant_id,
    load_deap_participant,
)

logger = logging.getLogger(__name__)

_CACHE_VERSION = "v2"
STIMULI_RESOLVED_PATH = DATA_DIR / "deap_stimuli_resolved.json"

CBRAMOD_SEGMENT_SAMPLES = CBRAMOD_SAMPLING_RATE * 4  # 800 samples at 200Hz


def trial_to_eeg_windows(
    trial_data: npt.NDArray[np.float32],
    *,
    source_segment_samples: int = SEGMENT_SAMPLES,
    target_segment_samples: int = CBRAMOD_SEGMENT_SAMPLES,
) -> npt.NDArray[np.float32]:
    """Slice a DEAP trial into non-overlapping 4s windows resampled for CBraMod.

    Shared by the training dataset and the runtime retrieval service so both
    sides produce windows with identical shape and preprocessing. `trial_data`
    is expected as (n_channels, n_samples) at DEAP's 128Hz sampling rate;
    returns (n_windows, n_channels, target_segment_samples) at 200Hz.
    """
    n_samples = trial_data.shape[1]
    n_segments = n_samples // source_segment_samples
    windows: list[npt.NDArray[np.float32]] = []
    for seg_idx in range(n_segments):
        start = seg_idx * source_segment_samples
        end = start + source_segment_samples
        segment = trial_data[:, start:end]  # (n_channels, source_segment_samples)
        resampled = resample(segment, target_segment_samples, axis=1).astype(np.float32)
        windows.append(resampled)
    if not windows:
        return np.zeros((0, trial_data.shape[0], target_segment_samples), dtype=np.float32)
    return np.stack(windows, axis=0)


def _audio_cache_key(resolved: list[dict[str, Any]], model_id: str) -> str:
    # Stable across hosts: the m4a basename is already a SHA-based content
    # hash of (normalized_artist, normalized_title) from audio_catalog, so
    # two files with the same basename have identical inputs. Deliberately
    # drops file mtime so Modal workers can reuse a cache built on a dev
    # machine after cache files are shipped in the image.
    parts = [_CACHE_VERSION, model_id]
    for entry in sorted(resolved, key=lambda r: r["trial_id"]):
        basename = Path(entry["audio_cache_path"]).name
        parts.append(f"{entry['trial_id']}:{basename}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _audio_cache_path(key: str) -> Path:
    return _cache_dir(DEAP_DATA_DIR) / f"clap_audio_{key}.npz"


def _resolve_audio_path(recorded_path: str) -> Path | None:
    """Map a recorded audio_cache_path to something that exists on this host.

    `fetch_deap_audio.py` writes the absolute path of each cached m4a into
    `deap_stimuli_resolved.json`, which is host-specific — a path like
    `/Users/foo/.../audio_cache/<sha>.m4a` won't exist on a Modal worker
    or another developer's machine. Since the filename is a content hash
    of `(normalized_artist, normalized_title)` the basename is globally
    unique, so we can fall back to `AUDIO_CACHE_DIR / basename` when the
    recorded absolute path is missing.
    """
    candidate = Path(recorded_path)
    if candidate.exists():
        return candidate
    fallback = AUDIO_CACHE_DIR / candidate.name
    if fallback.exists():
        return fallback
    return None


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

    Caches to `backend/data/deap/.cache/clap_audio_{hash}.npz`. The hash is
    host-portable (see `_audio_cache_key`) so cache files shipped in a Modal
    image hit on workers that never ran `fetch_deap_audio.py`.
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
        path = _resolve_audio_path(entry["audio_cache_path"])
        if path is None:
            logger.warning(f"skipping trial {entry['trial_id']}: audio file not found")
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
        *,
        subject_filter: list[int] | None = None,
        augment: bool = False,
        resolved_stimuli: list[dict[str, Any]] | None = None,
        audio_embeddings: dict[int, npt.NDArray[np.float32]] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else DEAP_DATA_DIR
        self.source_segment_samples = SEGMENT_SAMPLES  # 512 at 128Hz
        self.target_segment_samples = CBRAMOD_SEGMENT_SAMPLES  # 800 at 200Hz
        # Flag is plumbed but unused pending EEG-native augmentation; train/val
        # must remain separate instances so any future augmentation can't leak
        # across splits.
        self.augment = augment

        self.resolved_stimuli = resolved_stimuli or load_resolved_stimuli()
        self.audio_embeddings = audio_embeddings or build_audio_embedding_cache(self.resolved_stimuli)
        self.allowed_trial_ids = set(self.audio_embeddings.keys())

        if subject_filter is None:
            files = sorted(self.data_dir.glob("*.dat"))
        else:
            files = [self.data_dir / f"s{p:02d}.dat" for p in subject_filter]
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
            f"{len(files)} subjects, augment={self.augment}"
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
            windows = trial_to_eeg_windows(
                trial_data,
                source_segment_samples=self.source_segment_samples,
                target_segment_samples=self.target_segment_samples,
            )
            for window in windows:
                self.samples.append((window, trial_id, subject_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        segment, trial_id, subject_id = self.samples[idx]
        eeg = torch.from_numpy(segment)
        audio = torch.from_numpy(self.audio_embeddings[trial_id])
        return eeg, audio, trial_id, subject_id

    def subject_ids(self) -> list[int]:
        return sorted({s[2] for s in self.samples})
