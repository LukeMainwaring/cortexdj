"""Unit tests for ml.metrics — the shared classification metric surface."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from cortexdj.ml.metrics import (
    MajorityBaselinePredictor,
    balanced_accuracy,
    class_weights_from_labels,
    macro_f1,
    per_class_recall,
    prediction_counts,
)


class TestPerClassRecall:
    def test_perfect_predictions(self) -> None:
        y = np.array([0, 0, 1, 1])
        assert per_class_recall(y, y) == [1.0, 1.0]

    def test_constant_majority_predictor_on_skewed_labels(self) -> None:
        # 77% low, always predict low → recall(low)=1.0, recall(high)=0.0
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        y_pred = np.zeros(10, dtype=np.int64)
        assert per_class_recall(y_true, y_pred) == [1.0, 0.0]

    def test_absent_class_scores_zero_not_nan(self) -> None:
        # Val fold where a subject has 0 high-arousal trials
        y_true = np.zeros(10, dtype=np.int64)
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        recalls = per_class_recall(y_true, y_pred)
        assert recalls[0] == 0.8
        assert recalls[1] == 0.0  # no true class-1 samples — not NaN
        assert not any(np.isnan(recalls))


class TestBalancedAccuracy:
    def test_constant_predictor_scores_half(self) -> None:
        # Majority-class collapse on skewed labels should score 0.5 exactly.
        y_true = np.array([0] * 77 + [1] * 23)
        y_pred = np.zeros(100, dtype=np.int64)
        assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.5)

    def test_perfect_predictions(self) -> None:
        y = np.array([0, 1, 0, 1, 1])
        assert balanced_accuracy(y, y) == pytest.approx(1.0)


class TestMacroF1:
    def test_constant_predictor_below_half_on_skewed(self) -> None:
        # Macro F1 actively punishes collapse: one class has 0 F1.
        y_true = np.array([0] * 77 + [1] * 23)
        y_pred = np.zeros(100, dtype=np.int64)
        score = macro_f1(y_true, y_pred)
        assert 0.4 < score < 0.5  # f1(low) ~0.87, f1(high) = 0 → ~0.435

    def test_matches_manual_f1_on_balanced_case(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        # class 0: tp=1, fp=0, fn=1 → p=1.0, r=0.5, f1=0.667
        # class 1: tp=2, fp=1, fn=0 → p=0.667, r=1.0, f1=0.8
        # macro = (0.667 + 0.8) / 2 ≈ 0.7333
        assert macro_f1(y_true, y_pred) == pytest.approx(0.7333, abs=1e-3)

    def test_absent_class_treated_as_zero(self) -> None:
        y_true = np.zeros(10, dtype=np.int64)
        y_pred = np.zeros(10, dtype=np.int64)
        score = macro_f1(y_true, y_pred)
        # class 0: f1=1.0; class 1: undefined → 0.0; macro = 0.5
        assert score == pytest.approx(0.5)


class TestPredictionCounts:
    def test_both_classes_present(self) -> None:
        assert prediction_counts(np.array([0, 1, 1, 0, 1])) == [2, 3]

    def test_zero_padding_when_one_class_absent(self) -> None:
        assert prediction_counts(np.array([0, 0, 0])) == [3, 0]
        assert prediction_counts(np.array([1, 1, 1])) == [0, 3]


class TestClassWeightsFromLabels:
    def test_balanced_labels_yield_unit_weights(self) -> None:
        y = np.array([0, 0, 1, 1])
        weights = class_weights_from_labels(y)
        assert weights.tolist() == pytest.approx([1.0, 1.0])

    def test_skewed_labels_upweight_minority(self) -> None:
        # DEAP-like 77/23 skew: minority class gets much higher weight
        y = np.array([0] * 77 + [1] * 23)
        weights = class_weights_from_labels(y)
        # weight[low] = 100 / (2 * 77) ≈ 0.649; weight[high] = 100 / (2 * 23) ≈ 2.174
        assert weights[0].item() == pytest.approx(100 / (2 * 77), rel=1e-5)
        assert weights[1].item() == pytest.approx(100 / (2 * 23), rel=1e-5)
        assert weights[1] > weights[0]

    def test_empty_class_clamps_and_is_finite(self) -> None:
        # Degenerate fold where a class has zero samples — must not return inf
        y = np.array([0, 0, 0, 0, 0])
        weights = class_weights_from_labels(y)
        assert torch.isfinite(weights).all()
        # weight[high] = 5 / (2 * max(0, 1)) = 2.5
        assert weights[1].item() == pytest.approx(2.5)

    def test_empty_input_returns_ones(self) -> None:
        weights = class_weights_from_labels(np.array([], dtype=np.int64))
        assert weights.tolist() == [1.0, 1.0]

    def test_low_count_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        y = np.array([0] * 20 + [1] * 3)
        with caplog.at_level(logging.WARNING, logger="cortexdj.ml.metrics"):
            class_weights_from_labels(y)
        assert any("< 5 samples" in rec.message for rec in caplog.records)


class TestMajorityBaselinePredictor:
    def test_picks_modal_class(self) -> None:
        baseline = MajorityBaselinePredictor()
        baseline.fit(np.array([0, 0, 0, 1, 1]))
        assert baseline.majority_class == 0

    def test_predicts_constant_length(self) -> None:
        baseline = MajorityBaselinePredictor()
        baseline.fit(np.array([1, 1, 1, 0]))
        preds = baseline.predict(10)
        assert preds.shape == (10,)
        assert (preds == 1).all()

    def test_scores_half_on_balanced_accuracy(self) -> None:
        # The whole point: on skewed labels it beats accuracy but scores 0.5
        # on balanced accuracy — visibly dumb.
        baseline = MajorityBaselinePredictor()
        y_true = np.array([0] * 77 + [1] * 23)
        baseline.fit(y_true)
        y_pred = baseline.predict(len(y_true))
        assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.5)
