"""EEG dataset for emotion classification.

Loads DEAP data and extracts differential entropy features per segment
for training/inference.

DEAP preprocessed format: .dat pickle files per participant with:
  - data: (40, 40, 8064) — 40 trials, 40 channels (32 EEG + 8 peripheral), 8064 samples
  - labels: (40, 4) — [valence, arousal, dominance, liking] on 1-9 scale
  First 3 seconds (384 samples) are baseline; remaining 60s (7680 samples) are stimulus.
"""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from scipy.signal import resample
from torch.utils.data import Dataset

from cortexdj.ml.preprocessing import DEFAULT_SAMPLING_RATE, extract_features

logger = logging.getLogger(__name__)

# Bump when feature extraction or labeling logic changes.
_CACHE_VERSION = "v3"

NUM_EEG_CHANNELS = 32
SEGMENT_SAMPLES = DEFAULT_SAMPLING_RATE * 4  # 4-second segments (512 samples at 128Hz)
EMOTION_STATES = ["calm", "relaxed", "stressed", "excited"]
AROUSAL_THRESHOLD = 5.0
VALENCE_THRESHOLD = 5.0

# Label binarization strategy for DEAP's 1-9 Likert self-reports.
#
# `median_per_subject` (default): splits each axis at that subject's own
# median. Balanced within each subject, removes per-subject rating-scale bias.
#
# `median_global`: pooled median across all 32 participants × 40 trials.
# Roughly balanced labels but doesn't account for per-subject scale.
#
# `fixed_5`: threshold at >= 5. Produces ~25/75 skew — only useful for
# reproducing DEAP papers that adopted this convention.
LabelSplitStrategy = Literal["fixed_5", "median_global", "median_per_subject"]

# DEAP constants
DEAP_BASELINE_SAMPLES = 384  # 3 seconds at 128Hz

# CBraMod target sampling rate
CBRAMOD_SAMPLING_RATE = 200
CBRAMOD_SEGMENT_SAMPLES = CBRAMOD_SAMPLING_RATE * 4  # 800 samples for 4s at 200Hz


def scores_to_quadrant(arousal: float, valence: float) -> str:
    if arousal >= AROUSAL_THRESHOLD and valence >= VALENCE_THRESHOLD:
        return "excited"
    elif arousal < AROUSAL_THRESHOLD and valence >= VALENCE_THRESHOLD:
        return "relaxed"
    elif arousal >= AROUSAL_THRESHOLD and valence < VALENCE_THRESHOLD:
        return "stressed"
    else:
        return "calm"


def load_deap_participant(
    file_path: Path,
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
]:
    """Load a DEAP preprocessed .dat file.

    Returns EEG-only data with baseline stripped:
      - data: (40, 32, 7680) — 40 trials, 32 EEG channels, 60s at 128Hz
      - labels: (40, 4) — [valence, arousal, dominance, liking]
    """
    # DEAP is a trusted academic dataset that users explicitly download and place locally
    with open(file_path, "rb") as f:
        participant = pickle.load(f, encoding="latin1")  # noqa: S301

    data = participant["data"][:, :NUM_EEG_CHANNELS, DEAP_BASELINE_SAMPLES:]
    labels = participant["labels"]
    return data, labels


def _extract_participant_id(file_path: Path) -> int:
    """Extract participant number from filename like 's01.npz' or 's01.dat'."""
    return int(file_path.stem[1:])


def _compute_label_thresholds(files: list[Path], strategy: LabelSplitStrategy) -> dict[int, tuple[float, float]]:
    """Return (arousal_threshold, valence_threshold) per participant.

    For `fixed_5`, every subject gets (5.0, 5.0). For `median_global`,
    every subject gets the pooled median. For `median_per_subject`, each
    subject gets their own median computed across their own 40 trials.

    Note: the median strategies deserialize each `.dat` file twice on
    a cold cache build — once here (for labels only) and once in
    `_load_participant` (for labels + EEG data). This is intentional:
    thresholds must be known *before* segment labels are written, and
    the cold-cache path is fully amortized by the `.npz` cache on
    subsequent runs. Don't "optimize" this loop away.
    """
    if strategy == "fixed_5":
        return {_extract_participant_id(f): (AROUSAL_THRESHOLD, VALENCE_THRESHOLD) for f in files}

    per_subject_labels: dict[int, npt.NDArray[np.floating[Any]]] = {}
    for file_path in files:
        _data, labels = load_deap_participant(file_path)
        per_subject_labels[_extract_participant_id(file_path)] = labels

    if strategy == "median_global":
        all_labels = np.concatenate(list(per_subject_labels.values()), axis=0)
        valence_thr = float(np.median(all_labels[:, 0]))
        arousal_thr = float(np.median(all_labels[:, 1]))
        return {pid: (arousal_thr, valence_thr) for pid in per_subject_labels}

    # median_per_subject
    thresholds: dict[int, tuple[float, float]] = {}
    for pid, labels in per_subject_labels.items():
        thresholds[pid] = (float(np.median(labels[:, 1])), float(np.median(labels[:, 0])))
    return thresholds


def _cache_key(files: list[Path], mode: str, extra: str = "") -> str:
    """Build a cache key from source file paths and their mtimes."""
    parts = [_CACHE_VERSION, mode, extra]
    for f in sorted(files):
        parts.append(f"{f.name}:{f.stat().st_mtime_ns}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _cache_dir(data_dir: Path) -> Path:
    return data_dir / ".cache"


class DEAPFeatureDataset(Dataset[tuple[torch.Tensor, int, int]]):
    """DEAP dataset returning DE features for EEGNet training.

    The 3-second baseline period is stripped by load_deap_participant().
    Caches extracted features to disk for faster subsequent loads.
    """

    def __init__(
        self,
        data_dir: str | Path,
        participants: list[int] | None = None,
        segment_samples: int = SEGMENT_SAMPLES,
        label_split_strategy: LabelSplitStrategy = "median_per_subject",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.segment_samples = segment_samples
        self.label_split_strategy: LabelSplitStrategy = label_split_strategy
        self.samples: list[tuple[npt.NDArray[np.floating[Any]], int, int]] = []
        self.participant_ids: list[int] = []

        if participants is None:
            files = sorted(self.data_dir.glob("*.dat"))
        else:
            files = [self.data_dir / f"s{p:02d}.dat" for p in participants]
            files = [f for f in files if f.exists()]

        if not files:
            msg = (
                f"No DEAP .dat files found in {self.data_dir}. "
                "See backend/data/DEAP_SETUP.md for download instructions."
            )
            raise FileNotFoundError(msg)

        if self._try_load_cache(files):
            return

        thresholds = _compute_label_thresholds(files, label_split_strategy)
        for file_path in files:
            self._load_participant(file_path, thresholds)

        self._save_cache(files)

    def _cache_path(self, files: list[Path]) -> Path:
        key = _cache_key(files, "features", f"{self.segment_samples}_{self.label_split_strategy}")
        return _cache_dir(self.data_dir) / f"features_{key}.npz"

    def _try_load_cache(self, files: list[Path]) -> bool:
        path = self._cache_path(files)
        if not path.exists():
            return False
        try:
            data = np.load(path, allow_pickle=False)
            n = int(data["n_samples"])
            feature_dim = int(data["feature_dim"])
            all_features = data["features"].reshape(n, feature_dim)
            arousal_labels = data["arousal_labels"]
            valence_labels = data["valence_labels"]
            pid_array = data["participant_ids"]
            for i in range(n):
                self.samples.append((all_features[i], int(arousal_labels[i]), int(valence_labels[i])))
                self.participant_ids.append(int(pid_array[i]))
            logger.info(f"Loaded {n} cached feature segments from {path.name}")
            return True
        except Exception as e:
            logger.warning(f"Cache load failed, recomputing features: {e}")
            return False

    def _save_cache(self, files: list[Path]) -> None:
        path = self._cache_path(files)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self.samples)
        if n == 0:
            return
        feature_dim = self.samples[0][0].shape[0]
        all_features = np.array([s[0] for s in self.samples])
        arousal_labels = np.array([s[1] for s in self.samples])
        valence_labels = np.array([s[2] for s in self.samples])
        np.savez(
            path,
            features=all_features,
            arousal_labels=arousal_labels,
            valence_labels=valence_labels,
            participant_ids=np.array(self.participant_ids),
            n_samples=n,
            feature_dim=feature_dim,
        )
        logger.info(f"Cached {n} feature segments to {path.name}")

    def _load_participant(self, file_path: Path, thresholds: dict[int, tuple[float, float]]) -> None:
        participant_id = _extract_participant_id(file_path)
        arousal_thr, valence_thr = thresholds[participant_id]
        data, labels = load_deap_participant(file_path)
        n_trials = data.shape[0]

        for trial_idx in range(n_trials):
            trial_data = data[trial_idx]  # (32, 7680)
            valence = float(labels[trial_idx, 0])
            arousal = float(labels[trial_idx, 1])

            arousal_label = 1 if arousal >= arousal_thr else 0
            valence_label = 1 if valence >= valence_thr else 0

            n_samples = trial_data.shape[1]
            n_segments = n_samples // self.segment_samples

            for seg_idx in range(n_segments):
                start = seg_idx * self.segment_samples
                end = start + self.segment_samples
                segment = trial_data[:, start:end]

                features = extract_features(segment)
                self.samples.append((features, arousal_label, valence_label))
                self.participant_ids.append(participant_id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        features, arousal_label, valence_label = self.samples[idx]
        return torch.tensor(features, dtype=torch.float32), arousal_label, valence_label

    def get_labels(self, indices: list[int] | None = None) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Return `(arousal_labels, valence_labels)` as int64 arrays, optionally restricted to `indices`."""
        return _extract_labels(self.samples, indices)


class DEAPRawDataset(Dataset[tuple[torch.Tensor, int, int]]):
    """DEAP dataset returning raw EEG segments for pretrained model training.

    Resamples from 128Hz to 200Hz (CBraMod target) and returns raw
    (n_channels, n_times) tensors instead of DE features.
    Caches resampled segments to disk for faster subsequent loads.
    """

    def __init__(
        self,
        data_dir: str | Path,
        participants: list[int] | None = None,
        target_sfreq: int = CBRAMOD_SAMPLING_RATE,
        label_split_strategy: LabelSplitStrategy = "median_per_subject",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.target_sfreq = target_sfreq
        self.source_segment_samples = SEGMENT_SAMPLES  # 512 at 128Hz
        self.target_segment_samples = target_sfreq * 4  # 800 at 200Hz
        self.label_split_strategy: LabelSplitStrategy = label_split_strategy
        self.samples: list[tuple[npt.NDArray[np.floating[Any]], int, int]] = []
        self.participant_ids: list[int] = []

        if participants is None:
            files = sorted(self.data_dir.glob("*.dat"))
        else:
            files = [self.data_dir / f"s{p:02d}.dat" for p in participants]
            files = [f for f in files if f.exists()]

        if not files:
            msg = (
                f"No DEAP .dat files found in {self.data_dir}. "
                "See backend/data/DEAP_SETUP.md for download instructions."
            )
            raise FileNotFoundError(msg)

        if self._try_load_cache(files):
            return

        thresholds = _compute_label_thresholds(files, label_split_strategy)
        for file_path in files:
            self._load_participant(file_path, thresholds)

        self._save_cache(files)

    def _cache_path(self, files: list[Path]) -> Path:
        key = _cache_key(files, "raw", f"{self.target_sfreq}_{self.label_split_strategy}")
        return _cache_dir(self.data_dir) / f"raw_{key}.npz"

    def _try_load_cache(self, files: list[Path]) -> bool:
        path = self._cache_path(files)
        if not path.exists():
            return False
        try:
            data = np.load(path, allow_pickle=False)
            n = int(data["n_samples"])
            n_channels = int(data["n_channels"])
            n_times = int(data["n_times"])
            all_segments = data["segments"].reshape(n, n_channels, n_times)
            arousal_labels = data["arousal_labels"]
            valence_labels = data["valence_labels"]
            pid_array = data["participant_ids"]
            for i in range(n):
                self.samples.append((all_segments[i], int(arousal_labels[i]), int(valence_labels[i])))
                self.participant_ids.append(int(pid_array[i]))
            logger.info(f"Loaded {n} cached raw segments from {path.name}")
            return True
        except Exception as e:
            logger.warning(f"Cache load failed, recomputing raw segments: {e}")
            return False

    def _save_cache(self, files: list[Path]) -> None:
        path = self._cache_path(files)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self.samples)
        if n == 0:
            return
        seg0 = self.samples[0][0]
        all_segments = np.array([s[0] for s in self.samples])
        arousal_labels = np.array([s[1] for s in self.samples])
        valence_labels = np.array([s[2] for s in self.samples])
        np.savez(
            path,
            segments=all_segments,
            arousal_labels=arousal_labels,
            valence_labels=valence_labels,
            participant_ids=np.array(self.participant_ids),
            n_samples=n,
            n_channels=seg0.shape[0],
            n_times=seg0.shape[1],
        )
        logger.info(f"Cached {n} raw segments to {path.name}")

    def _load_participant(self, file_path: Path, thresholds: dict[int, tuple[float, float]]) -> None:
        participant_id = _extract_participant_id(file_path)
        arousal_thr, valence_thr = thresholds[participant_id]
        data, labels = load_deap_participant(file_path)
        n_trials = data.shape[0]

        for trial_idx in range(n_trials):
            trial_data = data[trial_idx]  # (32, 7680)
            valence = float(labels[trial_idx, 0])
            arousal = float(labels[trial_idx, 1])

            arousal_label = 1 if arousal >= arousal_thr else 0
            valence_label = 1 if valence >= valence_thr else 0

            n_samples = trial_data.shape[1]
            n_segments = n_samples // self.source_segment_samples

            for seg_idx in range(n_segments):
                start = seg_idx * self.source_segment_samples
                end = start + self.source_segment_samples
                segment = trial_data[:, start:end]  # (32, 512)

                # Resample 128Hz -> target (200Hz for CBraMod)
                resampled = resample(segment, self.target_segment_samples, axis=1)
                self.samples.append((resampled, arousal_label, valence_label))
                self.participant_ids.append(participant_id)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        segment, arousal_label, valence_label = self.samples[idx]
        return torch.tensor(segment, dtype=torch.float32), arousal_label, valence_label

    def get_labels(self, indices: list[int] | None = None) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Return `(arousal_labels, valence_labels)` as int64 arrays, optionally restricted to `indices`."""
        return _extract_labels(self.samples, indices)


def _extract_labels(
    samples: list[tuple[npt.NDArray[np.floating[Any]], int, int]],
    indices: list[int] | None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Shared implementation for `DEAP*Dataset.get_labels`."""
    if indices is None:
        arousal = np.fromiter((s[1] for s in samples), dtype=np.int64, count=len(samples))
        valence = np.fromiter((s[2] for s in samples), dtype=np.int64, count=len(samples))
    else:
        n = len(indices)
        arousal = np.fromiter((samples[i][1] for i in indices), dtype=np.int64, count=n)
        valence = np.fromiter((samples[i][2] for i in indices), dtype=np.int64, count=n)
    return arousal, valence


def load_dataset(
    mode: Literal["features", "raw"] = "features",
    data_dir: Path | None = None,
    participants: list[int] | None = None,
    label_split_strategy: LabelSplitStrategy = "median_per_subject",
) -> DEAPFeatureDataset | DEAPRawDataset:
    """Factory function for loading EEG datasets.

    Args:
        mode: "features" for DE feature vectors (EEGNet), "raw" for raw EEG (pretrained models).
        data_dir: Override default data directory.
        participants: List of participant IDs to load (None = all).
        label_split_strategy: How to binarize the 1-9 Likert labels. See
            `LabelSplitStrategy` for tradeoffs. Default `median_per_subject`
            gives roughly balanced labels per subject; `fixed_5` is an
            opt-in escape hatch for reproducing papers that used the
            >= 5 threshold.
    """
    from cortexdj.core.paths import DEAP_DATA_DIR

    resolved_dir = data_dir or DEAP_DATA_DIR
    if mode == "raw":
        return DEAPRawDataset(
            resolved_dir,
            participants=participants,
            label_split_strategy=label_split_strategy,
        )
    return DEAPFeatureDataset(
        resolved_dir,
        participants=participants,
        label_split_strategy=label_split_strategy,
    )
