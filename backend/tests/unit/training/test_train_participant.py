"""Tests for training one participant's quadrant classifiers on the made-up participant."""

from pathlib import Path

import numpy as np
import pytest

from cortexdj.ml.checkpoint import ParticipantModels
from cortexdj.training.train import train_participant
from tests.factories import FakeDeapParticipant, write_deap_participant, write_participant_ratings

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


class TestTrainingRaises:
    def test_a_fold_whose_training_trials_lack_a_low_valence_raises(self, tmp_path: Path) -> None:
        valence_arousal = np.column_stack([np.full(10, 7.0), np.tile([3.0, 7.0], 5)])
        valence_arousal[0, 0] = 2.0  # the one low-valence trial: its fold's training trials have none

        write_deap_participant(tmp_path, participant_id=1, n_trials=10, valence_arousal=valence_arousal)

        with pytest.raises(ValueError, match="valence"):
            train_participant(1, tmp_path)
