"""Classification metrics for EEG dual-head training.

Shared by the training loop, smoke tests, and `compare-models`. Provides
balanced accuracy, macro-F1, per-class recall, and class weighting without
an sklearn dependency — numpy + torch are already in the hot path and
these metrics are trivially derivable from a confusion matrix.

Macro-F1 is the headline metric: unlike raw accuracy, it scores a
constant predictor at ≤0.50, making majority-class collapse visible.
"""

import logging

import numpy as np
import numpy.typing as npt
import torch

logger = logging.getLogger(__name__)

# Below this many samples in a class within a training fold, the computed
# class weight becomes large enough to destabilize training. Warn loudly
# when we cross this line.
_CLASS_COUNT_WARN_THRESHOLD = 5


def _confusion_counts(
    y_true: npt.NDArray[np.integer],
    y_pred: npt.NDArray[np.integer],
    num_classes: int,
) -> npt.NDArray[np.int64]:
    """Build a (num_classes, num_classes) confusion matrix: rows=true, cols=pred."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def per_class_recall(
    y_true: npt.NDArray[np.integer],
    y_pred: npt.NDArray[np.integer],
    num_classes: int = 2,
) -> list[float]:
    """Recall for each class; classes with zero true samples score 0.0.

    Matches sklearn's `recall_score(..., zero_division=0)` semantics so
    absent-class folds don't NaN-poison the aggregate.
    """
    cm = _confusion_counts(y_true, y_pred, num_classes)
    recalls: list[float] = []
    for c in range(num_classes):
        true_count = int(cm[c].sum())
        if true_count == 0:
            recalls.append(0.0)
            continue
        recalls.append(float(cm[c, c]) / float(true_count))
    return recalls


def _per_class_precision(
    cm: npt.NDArray[np.int64],
    num_classes: int,
) -> list[float]:
    precisions: list[float] = []
    for c in range(num_classes):
        pred_count = int(cm[:, c].sum())
        if pred_count == 0:
            precisions.append(0.0)
            continue
        precisions.append(float(cm[c, c]) / float(pred_count))
    return precisions


def balanced_accuracy(
    y_true: npt.NDArray[np.integer],
    y_pred: npt.NDArray[np.integer],
    num_classes: int = 2,
) -> float:
    """Mean of per-class recall. Scores a constant predictor at 1/num_classes."""
    return float(np.mean(per_class_recall(y_true, y_pred, num_classes)))


def macro_f1(
    y_true: npt.NDArray[np.integer],
    y_pred: npt.NDArray[np.integer],
    num_classes: int = 2,
) -> float:
    """Unweighted mean of per-class F1. Absent classes contribute 0.

    Matches sklearn's `f1_score(average="macro", zero_division=0)`.
    """
    cm = _confusion_counts(y_true, y_pred, num_classes)
    recalls = per_class_recall(y_true, y_pred, num_classes)
    precisions = _per_class_precision(cm, num_classes)
    f1s: list[float] = []
    for p, r in zip(precisions, recalls, strict=True):
        denom = p + r
        f1s.append(0.0 if denom == 0.0 else 2.0 * p * r / denom)
    return float(np.mean(f1s))


def prediction_counts(
    y_pred: npt.NDArray[np.integer],
    num_classes: int = 2,
) -> list[int]:
    """Count predictions per class. Zero-padded to `num_classes`."""
    counts = np.bincount(y_pred.astype(np.int64), minlength=num_classes)
    return [int(c) for c in counts[:num_classes]]


def class_weights_from_labels(
    labels: npt.NDArray[np.integer],
    num_classes: int = 2,
) -> torch.Tensor:
    """Inverse-frequency class weights for `nn.CrossEntropyLoss`.

    weight[c] = N / (num_classes * max(count[c], 1))

    Using `max(count, 1)` keeps the weight finite on degenerate folds where
    one class has zero samples (vs. returning `inf`). Logs a warning when
    any class has fewer than `_CLASS_COUNT_WARN_THRESHOLD` samples — a very
    high inverse weight on a near-empty class is a training-instability
    time bomb even when finite.
    """
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes)[:num_classes]
    n_total = int(counts.sum())
    if n_total == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    min_count = int(counts.min())
    if min_count < _CLASS_COUNT_WARN_THRESHOLD:
        logger.warning(
            "class_weights_from_labels: class counts %s contain a class with "
            "< %d samples — weights may destabilize training",
            counts.tolist(),
            _CLASS_COUNT_WARN_THRESHOLD,
        )

    weights = n_total / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


class MajorityBaselinePredictor:
    """Fits the most-common class on training labels and predicts it everywhere.

    The reference point for `compare-models`. Any trained model that can't
    beat this on macro-F1 is doing nothing.
    """

    majority_class: int

    def __init__(self) -> None:
        self.majority_class = 0

    def fit(self, y_train: npt.NDArray[np.integer], num_classes: int = 2) -> None:
        counts = np.bincount(y_train.astype(np.int64), minlength=num_classes)
        self.majority_class = int(counts.argmax())

    def predict(self, n: int) -> npt.NDArray[np.int64]:
        return np.full(n, self.majority_class, dtype=np.int64)
