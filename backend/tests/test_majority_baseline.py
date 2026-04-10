"""Tests for `_majority_baseline_metrics` — the reference row that makes
future collapses visible in `compare-models`.

If this row ever stops being a trustworthy reference point, the entire
diagnostic signal for future training collapse is gone. The tests here
pin its numerical behavior on synthetic label distributions with known
properties.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from cortexdj.ml.dataset import DEAPFeatureDataset
from cortexdj.ml.train import TrainingConfig, _majority_baseline_metrics


def _fake_dataset(
    arousal: npt.NDArray[np.int64],
    valence: npt.NDArray[np.int64],
    participant_ids: list[int],
) -> DEAPFeatureDataset:
    """Build a DEAPFeatureDataset without touching the filesystem.

    Bypasses `__init__` so we can populate just the attributes
    `_majority_baseline_metrics` + `make_loso_splits` read, without a
    DEAP .dat file anywhere in sight.
    """
    ds = object.__new__(DEAPFeatureDataset)
    # Features are never read by the majority baseline path, so we pack a
    # zero vector per sample just to keep the `samples` tuple shape honest.
    feature_stub = np.zeros((1,), dtype=np.float32)
    ds.samples = [(feature_stub, int(a), int(v)) for a, v in zip(arousal, valence, strict=True)]
    ds.participant_ids = participant_ids
    return ds


class TestMajorityBaselineMetrics:
    def test_balanced_labels_score_half_on_all_metrics(self) -> None:
        # 4 subjects, each with 10 samples at perfectly balanced 50/50.
        # MajorityBaselinePredictor will pick class 0 (argmax of ties); val
        # fold accuracy is 0.5, balanced acc is 0.5, macro-F1 is 0.333.
        n_subjects = 4
        per_subject = 10
        arousal = np.tile(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64), n_subjects)
        valence = arousal.copy()
        participant_ids = [pid for pid in range(1, n_subjects + 1) for _ in range(per_subject)]

        dataset = _fake_dataset(arousal, valence, participant_ids)
        config = TrainingConfig(cv_mode="loso", seed=None)
        result = _majority_baseline_metrics(dataset, config)

        assert result["arousal_acc"] == pytest.approx(0.5)
        assert result["valence_acc"] == pytest.approx(0.5)
        assert result["avg_acc"] == pytest.approx(0.5)
        assert result["arousal_balanced_acc"] == pytest.approx(0.5)
        assert result["arousal_macro_f1"] == pytest.approx(1 / 3)
        assert result["avg_macro_f1"] == pytest.approx(1 / 3)

    def test_skewed_labels_reproduce_known_pre_fix_numbers(self) -> None:
        # The pathology this row was built to reveal: each subject is 77% low
        # / 23% high. A majority-class predictor gets 0.77 accuracy but
        # 0.5 balanced accuracy and ~0.435 macro-F1 — the exact gap between
        # "looks fine" and "is broken" that hid the collapse for weeks.
        n_subjects = 4
        low_per_subj = 77
        high_per_subj = 23
        per_subject = np.concatenate([np.zeros(low_per_subj, dtype=np.int64), np.ones(high_per_subj, dtype=np.int64)])
        arousal = np.tile(per_subject, n_subjects)
        valence = arousal.copy()
        participant_ids = [pid for pid in range(1, n_subjects + 1) for _ in range(low_per_subj + high_per_subj)]

        dataset = _fake_dataset(arousal, valence, participant_ids)
        config = TrainingConfig(cv_mode="loso", seed=None)
        result = _majority_baseline_metrics(dataset, config)

        assert result["arousal_acc"] == pytest.approx(0.77, abs=1e-6)
        assert result["valence_acc"] == pytest.approx(0.77, abs=1e-6)
        # Balanced acc treats both classes equally → constant-0 predictor = 0.5
        assert result["arousal_balanced_acc"] == pytest.approx(0.5)
        # Macro-F1: f1(low) = 2 * 0.77 * 1.0 / (0.77 + 1.0) ≈ 0.870,
        # f1(high) = 0 (no high predictions) → macro ≈ 0.435
        assert result["arousal_macro_f1"] == pytest.approx(0.435, abs=1e-2)
        assert result["valence_macro_f1"] == pytest.approx(0.435, abs=1e-2)

    def test_arousal_and_valence_can_have_different_majorities(self) -> None:
        # Arousal is 70/30 low-majority; valence is 30/70 high-majority.
        # Majority baseline picks opposite classes per head — confirms the
        # two heads are actually being fit independently (not sharing state).
        n_subjects = 4
        per_subject = 100
        arousal_per = np.concatenate([np.zeros(70, dtype=np.int64), np.ones(30, dtype=np.int64)])
        valence_per = np.concatenate([np.zeros(30, dtype=np.int64), np.ones(70, dtype=np.int64)])
        arousal = np.tile(arousal_per, n_subjects)
        valence = np.tile(valence_per, n_subjects)
        participant_ids = [pid for pid in range(1, n_subjects + 1) for _ in range(per_subject)]

        dataset = _fake_dataset(arousal, valence, participant_ids)
        config = TrainingConfig(cv_mode="loso", seed=None)
        result = _majority_baseline_metrics(dataset, config)

        # Both heads score 0.7 on accuracy (majority fraction)
        assert result["arousal_acc"] == pytest.approx(0.7)
        assert result["valence_acc"] == pytest.approx(0.7)
        # Balanced acc is 0.5 on both (constant predictor)
        assert result["arousal_balanced_acc"] == pytest.approx(0.5)
        assert result["valence_balanced_acc"] == pytest.approx(0.5)

    def test_empty_class_val_fold_does_not_nan(self) -> None:
        # Regression: a val fold where the held-out subject happens to have
        # zero samples in one class used to produce NaN in per-class recall.
        # The metrics module now returns 0 instead.
        n_subjects = 3
        per_subject = 20
        # Subject 1: all low. Subjects 2-3: 50/50.
        arousal = np.concatenate(
            [
                np.zeros(per_subject, dtype=np.int64),
                np.tile(np.array([0, 1], dtype=np.int64), per_subject // 2),
                np.tile(np.array([0, 1], dtype=np.int64), per_subject // 2),
            ]
        )
        valence = arousal.copy()
        participant_ids = [pid for pid in range(1, n_subjects + 1) for _ in range(per_subject)]

        dataset = _fake_dataset(arousal, valence, participant_ids)
        config = TrainingConfig(cv_mode="loso", seed=None)
        result = _majority_baseline_metrics(dataset, config)

        # Every metric must be finite — no NaN slips through the aggregation.
        for key, value in result.items():
            assert np.isfinite(value), f"{key} is not finite: {value}"
