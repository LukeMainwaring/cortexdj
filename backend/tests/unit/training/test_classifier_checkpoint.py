"""Tests for writing the quadrant-classifier checkpoint with ``train-classifier`` and reading it back."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from cortexdj.ml.band_features import differential_entropy, windows_of
from cortexdj.ml.checkpoint import load_checkpoint
from cortexdj.ml.dataset import load_deap_participant
from cortexdj.ml.quadrant_classifier import QuadrantClassifier
from cortexdj.ml.recipe import RECIPE
from cortexdj.training.command import describe, train_classifier
from tests.factories import write_deap_participant


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data_dir = tmp_path_factory.mktemp("deap")
    for participant_id in (1, 2):
        write_deap_participant(data_dir, participant_id=participant_id, n_trials=20, seed=participant_id)
    return data_dir


def _held_out_logits(data_dir: Path, models: list[QuadrantClassifier], fold_of_trial: list[int]) -> np.ndarray:
    eeg, _ = load_deap_participant(data_dir / "s01.dat")
    inputs = torch.as_tensor(differential_entropy(windows_of(eeg)), dtype=torch.float32)
    with torch.no_grad():
        return np.stack([models[fold](inputs[trial]).numpy() for trial, fold in enumerate(fold_of_trial)])


class TestTrainClassifier:
    def test_save_then_load_gives_the_same_predictions(self, data_dir: Path, tmp_path: Path) -> None:
        path = tmp_path / "quadrant_classifier.pt"

        trained = train_classifier([1], data_dir, path).participants[1]
        loaded = load_checkpoint(path).participants[1]

        assert loaded.fold_of_trial == trained.fold_of_trial
        assert loaded.metrics == trained.metrics
        np.testing.assert_array_equal(
            _held_out_logits(data_dir, loaded.models, loaded.fold_of_trial),
            _held_out_logits(data_dir, trained.models, trained.fold_of_trial),
        )

    def test_a_subset_run_overwrites_the_file_with_only_its_participants(self, data_dir: Path, tmp_path: Path) -> None:
        path = tmp_path / "quadrant_classifier.pt"
        train_classifier([1, 2], data_dir, path)

        train_classifier([2], data_dir, path)

        loaded = load_checkpoint(path)
        assert list(loaded.participants) == [2]
        assert loaded.summary.mean.mean == loaded.participants[2].metrics.mean
        assert loaded.summary.mean.std == 0.0

    def test_the_summary_averages_participants_with_their_spread(self, data_dir: Path, tmp_path: Path) -> None:
        checkpoint = train_classifier([1, 2], data_dir, tmp_path / "quadrant_classifier.pt")
        valence = [checkpoint.participants[p].metrics.valence for p in (1, 2)]

        assert checkpoint.summary.valence.mean == pytest.approx(sum(valence) / 2)
        assert checkpoint.summary.valence.std == pytest.approx(abs(valence[0] - valence[1]) / 2)

    def test_describe_prints_the_three_figures_with_their_spread(self, data_dir: Path, tmp_path: Path) -> None:
        summary = train_classifier([1, 2], data_dir, tmp_path / "quadrant_classifier.pt").summary

        text = describe(summary, n_participants=2)

        for label, spread in (("arousal", summary.arousal), ("valence", summary.valence), ("mean", summary.mean)):
            assert f"{label}" in text
            assert f"{spread.mean:.3f} ± {spread.std:.3f}" in text


class TestLoadCheckpoint:
    def test_a_changed_recipe_raises_naming_the_setting(self, data_dir: Path, tmp_path: Path) -> None:
        path = tmp_path / "quadrant_classifier.pt"
        train_classifier([1], data_dir, path)

        with pytest.raises(ValueError, match="weight_decay"):
            load_checkpoint(path, recipe=replace(RECIPE, weight_decay=RECIPE.weight_decay * 2))

    def test_a_changed_band_raises_naming_the_bands(self, data_dir: Path, tmp_path: Path) -> None:
        path = tmp_path / "quadrant_classifier.pt"
        train_classifier([1], data_dir, path)

        with pytest.raises(ValueError, match="bands"):
            load_checkpoint(path, recipe=replace(RECIPE, bands=RECIPE.bands[:3]))

    def test_a_missing_file_raises_naming_train_classifier(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="train-classifier"):
            load_checkpoint(tmp_path / "quadrant_classifier.pt")
