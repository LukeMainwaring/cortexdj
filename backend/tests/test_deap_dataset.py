"""Tests for dataset loading, feature extraction, and cross-validation splits."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from cortexdj.ml.dataset import (
    CBRAMOD_SEGMENT_SAMPLES,
    DEAP_BASELINE_SAMPLES,
    NUM_EEG_CHANNELS,
    SEGMENT_SAMPLES,
    DEAPFeatureDataset,
    DEAPRawDataset,
    load_dataset,
    load_deap_participant,
)
from cortexdj.ml.preprocessing import FREQ_BANDS
from cortexdj.ml.train import make_grouped_splits, make_loso_splits

_rng = np.random.default_rng(42)

# DEAP format: 40 channels (32 EEG + 8 peripheral), 8064 samples per trial
_DEAP_CHANNELS = 40
_DEAP_SAMPLES = 8064  # 384 baseline + 7680 stimulus at 128Hz
_N_TRIALS = 4  # Small for tests


def _make_mock_deap_file(path: Path, participant_id: int, n_trials: int = _N_TRIALS) -> None:
    """Create a mock DEAP .dat pickle file."""
    data = _rng.standard_normal((n_trials, _DEAP_CHANNELS, _DEAP_SAMPLES)).astype(np.float32)
    labels = _rng.uniform(1, 9, (n_trials, 4)).astype(np.float32)
    with open(path, "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)


@pytest.fixture
def deap_dir(tmp_path: Path) -> Path:
    d = tmp_path / "deap"
    d.mkdir()
    for pid in range(1, 4):
        _make_mock_deap_file(d / f"s{pid:02d}.dat", pid)
    return d


class TestLoadDeapParticipant:
    def test_output_shapes(self, deap_dir: Path) -> None:
        data, labels = load_deap_participant(deap_dir / "s01.dat")
        assert data.shape == (_N_TRIALS, NUM_EEG_CHANNELS, _DEAP_SAMPLES - DEAP_BASELINE_SAMPLES)
        assert labels.shape == (_N_TRIALS, 4)

    def test_baseline_stripped(self, deap_dir: Path) -> None:
        data, _ = load_deap_participant(deap_dir / "s01.dat")
        # After stripping 384 baseline samples from 8064 total
        assert data.shape[2] == 7680

    def test_only_eeg_channels(self, deap_dir: Path) -> None:
        data, _ = load_deap_participant(deap_dir / "s01.dat")
        assert data.shape[1] == NUM_EEG_CHANNELS  # 32, not 40


class TestDEAPFeatureDataset:
    def test_loads_all_participants(self, deap_dir: Path) -> None:
        ds = DEAPFeatureDataset(deap_dir)
        n_segments_per_trial = 7680 // SEGMENT_SAMPLES  # 15
        expected = 3 * _N_TRIALS * n_segments_per_trial
        assert len(ds) == expected

    def test_feature_shape(self, deap_dir: Path) -> None:
        ds = DEAPFeatureDataset(deap_dir)
        features, arousal, valence = ds[0]
        assert features.shape == (NUM_EEG_CHANNELS * len(FREQ_BANDS),)  # 160
        assert features.dtype == torch.float32

    def test_binary_labels(self, deap_dir: Path) -> None:
        ds = DEAPFeatureDataset(deap_dir)
        for i in range(min(10, len(ds))):
            _, arousal, valence = ds[i]
            assert arousal in (0, 1)
            assert valence in (0, 1)

    def test_participant_ids_tracked(self, deap_dir: Path) -> None:
        ds = DEAPFeatureDataset(deap_dir)
        assert len(ds.participant_ids) == len(ds)
        assert set(ds.participant_ids) == {1, 2, 3}

    def test_specific_participants(self, deap_dir: Path) -> None:
        ds = DEAPFeatureDataset(deap_dir, participants=[1, 2])
        assert set(ds.participant_ids) == {1, 2}

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="DEAP_SETUP"):
            DEAPFeatureDataset(tmp_path / "nonexistent")


class TestDEAPRawDataset:
    def test_loads_and_resamples(self, deap_dir: Path) -> None:
        ds = DEAPRawDataset(deap_dir)
        segment, arousal, valence = ds[0]
        assert segment.shape == (NUM_EEG_CHANNELS, CBRAMOD_SEGMENT_SAMPLES)  # (32, 800)
        assert segment.dtype == torch.float32

    def test_segment_count(self, deap_dir: Path) -> None:
        ds = DEAPRawDataset(deap_dir)
        n_segments_per_trial = 7680 // SEGMENT_SAMPLES  # 15
        expected = 3 * _N_TRIALS * n_segments_per_trial
        assert len(ds) == expected

    def test_participant_ids_tracked(self, deap_dir: Path) -> None:
        ds = DEAPRawDataset(deap_dir)
        assert len(ds.participant_ids) == len(ds)
        assert set(ds.participant_ids) == {1, 2, 3}


class TestLoadDatasetFactory:
    def test_features(self, deap_dir: Path) -> None:
        ds = load_dataset(mode="features", data_dir=deap_dir)
        assert isinstance(ds, DEAPFeatureDataset)

    def test_raw(self, deap_dir: Path) -> None:
        ds = load_dataset(mode="raw", data_dir=deap_dir)
        assert isinstance(ds, DEAPRawDataset)

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset(data_dir=tmp_path / "missing")


class TestLOSOSplits:
    def test_no_participant_overlap(self, deap_dir: Path) -> None:
        ds = DEAPFeatureDataset(deap_dir)
        splits = make_loso_splits(ds)
        for train_idx, val_idx in splits:
            train_pids = {ds.participant_ids[i] for i in train_idx}
            val_pids = {ds.participant_ids[i] for i in val_idx}
            assert train_pids.isdisjoint(val_pids)

    def test_all_samples_covered(self, deap_dir: Path) -> None:
        ds = DEAPFeatureDataset(deap_dir)
        splits = make_loso_splits(ds)
        all_val = set()
        for _, val_idx in splits:
            all_val.update(val_idx)
        assert all_val == set(range(len(ds)))

    def test_one_fold_per_participant(self, deap_dir: Path) -> None:
        ds = DEAPFeatureDataset(deap_dir)
        splits = make_loso_splits(ds)
        assert len(splits) == 3  # 3 participants

    def test_max_folds_limits(self, deap_dir: Path) -> None:
        ds = DEAPFeatureDataset(deap_dir)
        splits = make_loso_splits(ds, max_folds=2)
        assert len(splits) == 2


class TestGroupedSplits:
    def test_no_participant_overlap(self, deap_dir: Path) -> None:
        ds = DEAPFeatureDataset(deap_dir)
        splits = make_grouped_splits(ds, n_folds=3)
        for train_idx, val_idx in splits:
            train_pids = {ds.participant_ids[i] for i in train_idx}
            val_pids = {ds.participant_ids[i] for i in val_idx}
            assert train_pids.isdisjoint(val_pids)

    def test_correct_fold_count(self, deap_dir: Path) -> None:
        ds = DEAPFeatureDataset(deap_dir)
        splits = make_grouped_splits(ds, n_folds=3)
        assert len(splits) == 3
