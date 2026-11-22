"""DEAP loading and the constants the EEG pipelines share.

DEAP preprocessed format: .dat pickle files per participant with:
  - data: (40, 40, 8064) — 40 trials, 40 channels (32 EEG + 8 peripheral), 8064 samples
  - labels: (40, 4) — [valence, arousal, dominance, liking] on 1-9 scale; the Kaggle mirror's
    valence and arousal are altered, so those come from ``load_participant_ratings``
  First 3 seconds (384 samples) are baseline; remaining 60s (7680 samples) are stimulus.
"""

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

NUM_EEG_CHANNELS = 32
DEAP_SAMPLING_RATE = 128
SEGMENT_SAMPLES = DEAP_SAMPLING_RATE * 4  # 4-second segments (512 samples at 128Hz)

# Multiply DEAP µV signals by this before feeding CBraMod. Without it, CBraMod
# sees signals ~100× too large and its pretrained features operate
# off-distribution. Derivation: NeuralBench's CBraMod recipe (neuralbench-repo
# `neuralbench/models/cbramod.yaml`: scaler=null, scale_factor=10000) applies
# 1e4 to MNE volts; the µV equivalent is µV × 1e-6 V/µV × 1e4 = µV × 1e-2.
CBRAMOD_SCALE_FACTOR = 0.01

# DEAP constants
DEAP_BASELINE_SAMPLES = 384  # 3 seconds at 128Hz

# DEAP's published ratings, kept beside the .dat files (gitignored: DEAP's licence forbids
# redistribution). The only trusted source of valence and arousal.
RATINGS_FILENAME = "participant_ratings.csv"

# CBraMod target sampling rate
CBRAMOD_SAMPLING_RATE = 200
CBRAMOD_SEGMENT_SAMPLES = CBRAMOD_SAMPLING_RATE * 4  # 800 samples for 4s at 200Hz


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


HIGH_RATING = 5.0


def is_high(rating: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Whether a 1-9 self-rating counts as high: at or above 5, the same line for every participant."""
    return np.asarray(rating) >= HIGH_RATING


@dataclass(frozen=True)
class ParticipantRatings:
    """A DEAP participant's published self-ratings, one per trial in the EEG file's trial order."""

    valence: npt.NDArray[np.float64]
    arousal: npt.NDArray[np.float64]


def load_participant_ratings(
    participant_id: int, data_dir: Path, *, eeg_labels: npt.NDArray[np.floating[Any]] | None = None
) -> ParticipantRatings:
    """Read a participant's valence and arousal from ``participant_ratings.csv`` in ``data_dir``.

    Rows join on participant and experiment id: trial index ``e`` in ``sNN.dat`` is
    ``Experiment_id`` ``e + 1``. The file's rows are in presentation order, not EEG order.
    Do not read valence or arousal from the ``.dat`` label array: the Kaggle mirror's copy
    is altered (``9 - rating`` in every trial where both ratings exceed 5).

    The ratings are checked against ``sNN.dat``'s label array; a caller that has already loaded
    the file passes that array as ``eeg_labels`` so it is not unpickled a second time.
    """
    ratings_path = data_dir / RATINGS_FILENAME
    if not ratings_path.exists():
        raise FileNotFoundError(
            f"No {RATINGS_FILENAME} in {data_dir} for participant {participant_id}; "
            "backend/data/DEAP_SETUP.md says where to get it"
        )
    with ratings_path.open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if int(row["Participant_id"]) == participant_id]
    by_trial = {int(row["Experiment_id"]) - 1: row for row in rows}

    if eeg_labels is None:
        _, eeg_labels = load_deap_participant(data_dir / f"s{participant_id:02d}.dat")
    n_trials = len(eeg_labels)
    if sorted(by_trial) != list(range(n_trials)):
        raise ValueError(
            f"{RATINGS_FILENAME} has experiment ids {sorted(e + 1 for e in by_trial)} for participant "
            f"{participant_id}, but its EEG file holds {n_trials} trials"
        )
    ordered = [by_trial[e] for e in range(n_trials)]
    dominance_liking = np.array([[float(row["Dominance"]), float(row["Liking"])] for row in ordered])
    # The mirror's dominance and liking are DEAP's own, so they agreeing is the check that
    # trial e of the EEG file is the ratings row for experiment e + 1.
    if not np.allclose(dominance_liking, eeg_labels[:, 2:4], atol=1e-4):
        raise ValueError(
            f"{RATINGS_FILENAME}'s dominance and liking for participant {participant_id} differ from "
            "the EEG file's label array, so the ratings do not line up with its trials"
        )
    return ParticipantRatings(
        valence=np.array([float(row["Valence"]) for row in ordered]),
        arousal=np.array([float(row["Arousal"]) for row in ordered]),
    )
