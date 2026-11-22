"""Tests for training one participant's quadrant classifiers on the made-up participant."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from cortexdj.ml.checkpoint import ParticipantModels
from cortexdj.training.train import cross_validate, train_participant
from tests.factories import (
    FakeDeapParticipant,
    flatten_channel,
    write_deap_participant,
    write_participant_ratings,
)

_TRIALS = 20


@pytest.fixture(scope="module")
def planted_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("deap")


@pytest.fixture(scope="module")
def planted(planted_dir: Path) -> FakeDeapParticipant:
    return write_deap_participant(planted_dir, participant_id=1, n_trials=_TRIALS)


@pytest.fixture(scope="module")
def trained(planted_dir: Path, planted: FakeDeapParticipant) -> ParticipantModels:
    return train_participant(1, planted_dir)


class TestFolds:
    def test_every_trial_is_held_out_by_exactly_one_of_five_folds(self, trained: ParticipantModels) -> None:
        assert len(trained.fold_of_trial) == _TRIALS
        assert sorted(set(trained.fold_of_trial)) == [0, 1, 2, 3, 4]
        assert len(trained.models) == 5

    def test_each_quadrant_is_spread_across_the_folds(
        self, planted: FakeDeapParticipant, trained: ParticipantModels
    ) -> None:
        quadrants = (planted.arousal >= 5) * 2 + (planted.valence >= 5)
        folds = np.array(trained.fold_of_trial)

        for quadrant in np.unique(quadrants):
            per_fold = np.bincount(folds[quadrants == quadrant], minlength=5)
            assert per_fold.max() - per_fold.min() <= 1


class TestScores:
    def test_a_planted_signal_scores_well_above_chance(self, trained: ParticipantModels) -> None:
        assert trained.metrics.arousal > 0.8
        assert trained.metrics.valence > 0.8
        assert trained.metrics.mean == pytest.approx((trained.metrics.arousal + trained.metrics.valence) / 2)

    def test_shuffled_labels_score_near_chance(self, tmp_path: Path) -> None:
        written = write_deap_participant(tmp_path, participant_id=1, n_trials=40)
        shuffled = written.ratings.copy()
        shuffled[:, :2] = np.random.default_rng(1).permutation(shuffled[:, :2])
        write_participant_ratings(tmp_path, 1, shuffled)

        metrics = train_participant(1, tmp_path).metrics

        assert metrics.arousal == pytest.approx(0.5, abs=0.15)
        assert metrics.valence == pytest.approx(0.5, abs=0.15)

    def test_a_rerun_gives_the_same_metrics(self, planted_dir: Path, trained: ParticipantModels) -> None:
        rerun = train_participant(1, planted_dir)

        assert rerun.metrics == trained.metrics
        assert rerun.fold_of_trial == trained.fold_of_trial


class _PassThrough(nn.Module):
    """Its input is its logits; a zero-weighted parameter gives the optimizer something to hold."""

    def __init__(self) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0 * self.unused


class TestCrossValidateThreshold:
    def test_a_logit_of_exactly_zero_scores_as_high_as_seeding_reads_it(self) -> None:
        # Seeding reads sigmoid(0) = 0.5 as high, so the held-out metric must too: every high
        # trial's logit is exactly 0 and every low trial's is -1, a perfect read at >= 0.
        high = np.array([[True, False], [False, True]] * 5)
        windows = np.where(high, 0.0, -1.0)[:, None, :]  # (trials, 1 window, 2 "features")

        _, metrics = cross_validate(
            windows,
            high,
            np.arange(10) % 5,
            make_model=lambda _inputs: _PassThrough(),
            to_input=lambda w: w,
        )

        assert (metrics.arousal, metrics.valence) == (1.0, 1.0)


class TestTrainingRaises:
    def test_a_fold_whose_training_trials_lack_a_low_valence_raises(self, tmp_path: Path) -> None:
        valence_arousal = np.column_stack([np.full(10, 7.0), np.tile([3.0, 7.0], 5)])
        valence_arousal[0, 0] = 2.0  # the one low-valence trial: its fold's training trials have none

        write_deap_participant(tmp_path, participant_id=1, n_trials=10, valence_arousal=valence_arousal)

        with pytest.raises(ValueError, match="valence"):
            train_participant(1, tmp_path)

    def test_a_flat_eeg_channel_raises_rather_than_training_on_infinity(self, tmp_path: Path) -> None:
        written = write_deap_participant(tmp_path, participant_id=1, n_trials=10)
        flatten_channel(written.eeg_path, trial=2, channel=5)

        with pytest.raises(ValueError, match="non-finite"):
            train_participant(1, tmp_path)
