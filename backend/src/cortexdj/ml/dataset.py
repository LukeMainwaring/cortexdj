"""EEG dataset for emotion classification.

Loads synthetic or DEAP data and extracts differential entropy features
per segment for training/inference.

Synthetic data format: .npz files per participant with:
  - data: (n_trials, n_channels, n_samples)
  - labels: (n_trials, 4) — [valence, arousal, dominance, liking]

DEAP preprocessed format: .dat pickle files per participant with:
  - data: (40, 40, 8064) — 40 trials, 40 channels (32 EEG + 8 peripheral), 8064 samples
  - labels: (40, 4) — [valence, arousal, dominance, liking] on 1-9 scale
  First 3 seconds (384 samples) are baseline; remaining 60s (7680 samples) are stimulus.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from scipy.signal import resample
from torch.utils.data import Dataset

from cortexdj.ml.preprocessing import DEFAULT_SAMPLING_RATE, extract_features

NUM_EEG_CHANNELS = 32
SEGMENT_SAMPLES = DEFAULT_SAMPLING_RATE * 4  # 4-second segments (512 samples at 128Hz)
EMOTION_STATES = ["calm", "relaxed", "stressed", "excited"]
AROUSAL_THRESHOLD = 5.0
VALENCE_THRESHOLD = 5.0

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


def load_synthetic_participant(
    file_path: Path,
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
]:
    npz = np.load(file_path)
    return npz["data"], npz["labels"]


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


class EEGEmotionDataset(Dataset[tuple[torch.Tensor, int, int]]):
    """PyTorch dataset that loads EEG data and extracts DE features per segment.

    Each item yields (features_tensor, arousal_label, valence_label) where
    labels are binary (0=low, 1=high).
    """

    def __init__(
        self,
        data_dir: str | Path,
        participants: list[int] | None = None,
        segment_samples: int = SEGMENT_SAMPLES,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.segment_samples = segment_samples
        self.samples: list[tuple[npt.NDArray[np.floating[Any]], int, int]] = []
        self.participant_ids: list[int] = []

        if participants is None:
            files = sorted(self.data_dir.glob("*.npz"))
        else:
            files = [self.data_dir / f"s{p:02d}.npz" for p in participants]
            files = [f for f in files if f.exists()]

        for file_path in files:
            self._load_participant(file_path)

    def _load_participant(self, file_path: Path) -> None:
        participant_id = _extract_participant_id(file_path)
        data, labels = load_synthetic_participant(file_path)
        n_trials = data.shape[0]

        for trial_idx in range(n_trials):
            trial_data = data[trial_idx, :NUM_EEG_CHANNELS, :]  # (32, n_samples)
            valence = float(labels[trial_idx, 0])
            arousal = float(labels[trial_idx, 1])

            arousal_label = 1 if arousal >= AROUSAL_THRESHOLD else 0
            valence_label = 1 if valence >= VALENCE_THRESHOLD else 0

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


class DEAPFeatureDataset(Dataset[tuple[torch.Tensor, int, int]]):
    """DEAP dataset returning DE features for EEGNet training.

    Same interface as EEGEmotionDataset but loads from DEAP preprocessed .dat files.
    The 3-second baseline period is stripped by load_deap_participant().
    """

    def __init__(
        self,
        data_dir: str | Path,
        participants: list[int] | None = None,
        segment_samples: int = SEGMENT_SAMPLES,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.segment_samples = segment_samples
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

        for file_path in files:
            self._load_participant(file_path)

    def _load_participant(self, file_path: Path) -> None:
        participant_id = _extract_participant_id(file_path)
        data, labels = load_deap_participant(file_path)
        n_trials = data.shape[0]

        for trial_idx in range(n_trials):
            trial_data = data[trial_idx]  # (32, 7680)
            valence = float(labels[trial_idx, 0])
            arousal = float(labels[trial_idx, 1])

            arousal_label = 1 if arousal >= AROUSAL_THRESHOLD else 0
            valence_label = 1 if valence >= VALENCE_THRESHOLD else 0

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


class DEAPRawDataset(Dataset[tuple[torch.Tensor, int, int]]):
    """DEAP dataset returning raw EEG segments for pretrained model training.

    Resamples from 128Hz to 200Hz (CBraMod target) and returns raw
    (n_channels, n_times) tensors instead of DE features.
    """

    def __init__(
        self,
        data_dir: str | Path,
        participants: list[int] | None = None,
        target_sfreq: int = CBRAMOD_SAMPLING_RATE,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.target_sfreq = target_sfreq
        self.source_segment_samples = SEGMENT_SAMPLES  # 512 at 128Hz
        self.target_segment_samples = target_sfreq * 4  # 800 at 200Hz
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

        for file_path in files:
            self._load_participant(file_path)

    def _load_participant(self, file_path: Path) -> None:
        participant_id = _extract_participant_id(file_path)
        data, labels = load_deap_participant(file_path)
        n_trials = data.shape[0]

        for trial_idx in range(n_trials):
            trial_data = data[trial_idx]  # (32, 7680)
            valence = float(labels[trial_idx, 0])
            arousal = float(labels[trial_idx, 1])

            arousal_label = 1 if arousal >= AROUSAL_THRESHOLD else 0
            valence_label = 1 if valence >= VALENCE_THRESHOLD else 0

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


def load_dataset(
    source: Literal["synthetic", "deap"] = "synthetic",
    mode: Literal["features", "raw"] = "features",
    data_dir: Path | None = None,
    participants: list[int] | None = None,
) -> EEGEmotionDataset | DEAPFeatureDataset | DEAPRawDataset:
    """Factory function for loading EEG datasets.

    Args:
        source: Dataset source — "synthetic" or "deap".
        mode: "features" for DE feature vectors (EEGNet), "raw" for raw EEG (pretrained models).
        data_dir: Override default data directory.
        participants: List of participant IDs to load (None = all).
    """
    from cortexdj.core.paths import DEAP_DATA_DIR, SYNTHETIC_DATA_DIR

    if source == "synthetic":
        if mode == "raw":
            msg = "Raw mode is not supported for synthetic data. Use mode='features'."
            raise ValueError(msg)
        resolved_dir = data_dir or SYNTHETIC_DATA_DIR
        return EEGEmotionDataset(resolved_dir, participants=participants)

    # source == "deap"
    resolved_dir = data_dir or DEAP_DATA_DIR
    if mode == "raw":
        return DEAPRawDataset(resolved_dir, participants=participants)
    return DEAPFeatureDataset(resolved_dir, participants=participants)
