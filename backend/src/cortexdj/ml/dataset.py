"""EEG dataset for emotion classification.

Loads synthetic or DEAP data and extracts differential entropy features
per segment for training/inference.

Synthetic data format: .npz files per participant with:
  - data: (n_trials, n_channels, n_samples)
  - labels: (n_trials, 4) — [valence, arousal, dominance, liking]

DEAP data format: .dat pickle files per participant with:
  - data: (40, 40, 8064) — 40 trials, 40 channels, 8064 samples
  - labels: (40, 4) — [valence, arousal, dominance, liking]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cortexdj.ml.preprocessing import DEFAULT_SAMPLING_RATE, extract_features

NUM_EEG_CHANNELS = 32
SEGMENT_SAMPLES = DEFAULT_SAMPLING_RATE * 4  # 4-second segments (512 samples)
EMOTION_STATES = ["calm", "relaxed", "stressed", "excited"]
AROUSAL_THRESHOLD = 5.0
VALENCE_THRESHOLD = 5.0


def scores_to_quadrant(arousal: float, valence: float) -> str:
    """Map arousal/valence scores to emotion quadrant label."""
    if arousal >= AROUSAL_THRESHOLD and valence >= VALENCE_THRESHOLD:
        return "excited"
    elif arousal < AROUSAL_THRESHOLD and valence >= VALENCE_THRESHOLD:
        return "relaxed"
    elif arousal >= AROUSAL_THRESHOLD and valence < VALENCE_THRESHOLD:
        return "stressed"
    else:
        return "calm"


def load_synthetic_participant(file_path: Path) -> tuple[
    np.ndarray[tuple[int, ...], np.dtype[np.floating[object]]],
    np.ndarray[tuple[int, ...], np.dtype[np.floating[object]]],
]:
    """Load a single participant's synthetic data.

    Returns:
        (data, labels) where data is (n_trials, n_channels, n_samples)
        and labels is (n_trials, 4).
    """
    npz = np.load(file_path)
    return npz["data"], npz["labels"]


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
        self.samples: list[tuple[np.ndarray[tuple[int, ...], np.dtype[np.floating[object]]], int, int]] = []

        # Determine which participants to load
        if participants is None:
            files = sorted(self.data_dir.glob("*.npz"))
        else:
            files = [self.data_dir / f"s{p:02d}.npz" for p in participants]
            files = [f for f in files if f.exists()]

        for file_path in files:
            self._load_participant(file_path)

    def _load_participant(self, file_path: Path) -> None:
        """Load one participant's data and segment into training samples."""
        data, labels = load_synthetic_participant(file_path)
        n_trials = data.shape[0]

        for trial_idx in range(n_trials):
            trial_data = data[trial_idx, :NUM_EEG_CHANNELS, :]  # (32, n_samples)
            valence = float(labels[trial_idx, 0])
            arousal = float(labels[trial_idx, 1])

            arousal_label = 1 if arousal >= AROUSAL_THRESHOLD else 0
            valence_label = 1 if valence >= VALENCE_THRESHOLD else 0

            # Segment the trial into fixed-length windows
            n_samples = trial_data.shape[1]
            n_segments = n_samples // self.segment_samples

            for seg_idx in range(n_segments):
                start = seg_idx * self.segment_samples
                end = start + self.segment_samples
                segment = trial_data[:, start:end]

                features = extract_features(segment)
                self.samples.append((features, arousal_label, valence_label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        features, arousal_label, valence_label = self.samples[idx]
        return torch.tensor(features, dtype=torch.float32), arousal_label, valence_label
