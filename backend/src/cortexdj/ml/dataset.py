"""DEAP loading and the constants the EEG pipelines share.

DEAP preprocessed format: .dat pickle files per participant with:
  - data: (40, 40, 8064) — 40 trials, 40 channels (32 EEG + 8 peripheral), 8064 samples
  - labels: (40, 4) — [valence, arousal, dominance, liking] on 1-9 scale
  First 3 seconds (384 samples) are baseline; remaining 60s (7680 samples) are stimulus.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from cortexdj.ml.preprocessing import DEFAULT_SAMPLING_RATE

NUM_EEG_CHANNELS = 32
SEGMENT_SAMPLES = DEFAULT_SAMPLING_RATE * 4  # 4-second segments (512 samples at 128Hz)
AROUSAL_THRESHOLD = 5.0
VALENCE_THRESHOLD = 5.0

# Multiply DEAP µV signals by this before feeding CBraMod. Without it, CBraMod
# sees signals ~100× too large and its pretrained features operate
# off-distribution. Derivation: NeuralBench's CBraMod recipe (neuralbench-repo
# `neuralbench/models/cbramod.yaml`: scaler=null, scale_factor=10000) applies
# 1e4 to MNE volts; the µV equivalent is µV × 1e-6 V/µV × 1e4 = µV × 1e-2.
CBRAMOD_SCALE_FACTOR = 0.01

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
