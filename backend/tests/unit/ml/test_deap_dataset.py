"""Tests for loading a DEAP participant file."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from cortexdj.ml.dataset import DEAP_BASELINE_SAMPLES, NUM_EEG_CHANNELS, load_deap_participant

_rng = np.random.default_rng(42)

# DEAP format: 40 channels (32 EEG + 8 peripheral), 8064 samples per trial
_DEAP_CHANNELS = 40
_DEAP_SAMPLES = 8064  # 384 baseline + 7680 stimulus at 128Hz
_N_TRIALS = 4  # Small for tests


def _make_mock_deap_file(path: Path) -> None:
    """Create a mock DEAP .dat pickle file."""
    data = _rng.standard_normal((_N_TRIALS, _DEAP_CHANNELS, _DEAP_SAMPLES)).astype(np.float32)
    labels = _rng.uniform(1, 9, (_N_TRIALS, 4)).astype(np.float32)
    with open(path, "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)


@pytest.fixture
def deap_dir(tmp_path: Path) -> Path:
    d = tmp_path / "deap"
    d.mkdir()
    _make_mock_deap_file(d / "s01.dat")
    return d


class TestLoadDeapParticipant:
    def test_output_shapes(self, deap_dir: Path) -> None:
        data, labels = load_deap_participant(deap_dir / "s01.dat")
        assert data.shape == (_N_TRIALS, NUM_EEG_CHANNELS, _DEAP_SAMPLES - DEAP_BASELINE_SAMPLES)
        assert labels.shape == (_N_TRIALS, 4)
