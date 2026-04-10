"""Training loop smoke test.

Would have caught the DEAP majority-class collapse on day one: trains
`train_fold_eegnet` for a handful of epochs on a tiny balanced synthetic
dataset where each head's label is a clean linear function of one input
feature, then asserts that both classes are actually being predicted and
that the two heads aren't accidentally tied to each other.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cortexdj.ml.metrics import class_weights_from_labels, per_class_recall, prediction_counts
from cortexdj.ml.model import FEATURE_DIM
from cortexdj.ml.train import TrainingConfig, train_fold_eegnet


class _SyntheticFeatureDataset(Dataset[tuple[torch.Tensor, int, int]]):
    """Linearly separable DE-feature lookalike for arousal and valence.

    Arousal labels are balanced 50/50; valence labels are skewed 70/30 so
    the two heads can't accidentally reuse each other's answer. Features
    are Gaussian noise with a strong signal on dimension 0 (arousal) and
    dimension 1 (valence).
    """

    def __init__(self, n: int = 256, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.features = rng.standard_normal((n, FEATURE_DIM)).astype(np.float32) * 0.1
        arousal_labels = (np.arange(n) % 2).astype(np.int64)  # 50/50
        # 70/30 valence: build the exact ratio, then shuffle in place so
        # arousal and valence labels are uncorrelated at the sample level.
        # (`rng.shuffle` is a permutation, so the 70/30 count is preserved.)
        split = int(n * 0.7)
        valence_labels = np.concatenate(
            [np.zeros(split, dtype=np.int64), np.ones(n - split, dtype=np.int64)]
        )
        rng.shuffle(valence_labels)

        # Plant a clean linear signal on feature[0] and feature[1].
        self.features[:, 0] += (arousal_labels * 2 - 1) * 3.0
        self.features[:, 1] += (valence_labels * 2 - 1) * 3.0

        self.arousal = arousal_labels
        self.valence = valence_labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        return (
            torch.from_numpy(self.features[idx]),
            int(self.arousal[idx]),
            int(self.valence[idx]),
        )


def _collect_val_predictions(
    model: torch.nn.Module, val_loader: DataLoader[tuple[torch.Tensor, int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    arousal_true: list[int] = []
    arousal_pred: list[int] = []
    valence_true: list[int] = []
    valence_pred: list[int] = []
    with torch.inference_mode():
        for features, a, v in val_loader:
            arousal_logits, valence_logits = model(features)
            arousal_true.extend(a.tolist())
            arousal_pred.extend(arousal_logits.argmax(1).tolist())
            valence_true.extend(v.tolist())
            valence_pred.extend(valence_logits.argmax(1).tolist())
    return (
        np.asarray(arousal_true, dtype=np.int64),
        np.asarray(arousal_pred, dtype=np.int64),
        np.asarray(valence_true, dtype=np.int64),
        np.asarray(valence_pred, dtype=np.int64),
    )


def test_train_fold_eegnet_does_not_collapse_to_constant() -> None:
    train_dataset = _SyntheticFeatureDataset(n=256, seed=1)
    val_dataset = _SyntheticFeatureDataset(n=64, seed=2)
    train_loader: DataLoader[tuple[torch.Tensor, int, int]] = DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader: DataLoader[tuple[torch.Tensor, int, int]] = DataLoader(
        val_dataset, batch_size=32, shuffle=False
    )
    config = TrainingConfig(
        model_type="eegnet",
        epochs=8,
        lr=1e-2,
        batch_size=32,
        patience=8,
        seed=42,
        no_early_stop=True,
    )
    device = torch.device("cpu")

    # Build criteria the same way the real fold loop does — directly from
    # labels, no loader iteration.
    arousal_weights = class_weights_from_labels(train_dataset.arousal).to(device)
    valence_weights = class_weights_from_labels(train_dataset.valence).to(device)
    arousal_criterion = nn.CrossEntropyLoss(
        weight=arousal_weights, label_smoothing=config.label_smoothing
    )
    valence_criterion = nn.CrossEntropyLoss(
        weight=valence_weights, label_smoothing=config.label_smoothing
    )

    # Force CPU to keep the test fast and hermetic across runners. The
    # fold function takes `device` explicitly, so no monkeypatching needed.
    model, _metrics = train_fold_eegnet(
        train_loader,
        val_loader,
        arousal_criterion=arousal_criterion,
        valence_criterion=valence_criterion,
        config=config,
        device=device,
    )

    a_true, a_pred, v_true, v_pred = _collect_val_predictions(model, val_loader)
    arousal_recalls = per_class_recall(a_true, a_pred)
    valence_recalls = per_class_recall(v_true, v_pred)
    arousal_counts = prediction_counts(a_pred)
    valence_counts = prediction_counts(v_pred)

    # 1. Neither head collapsed to a constant: both classes predicted on both heads.
    assert min(arousal_counts) > 0, f"arousal collapsed: {arousal_counts}"
    assert min(valence_counts) > 0, f"valence collapsed: {valence_counts}"

    # 2. Both classes have non-trivial recall on both heads.
    assert all(r > 0.1 for r in arousal_recalls), f"arousal recalls too low: {arousal_recalls}"
    assert all(r > 0.1 for r in valence_recalls), f"valence recalls too low: {valence_recalls}"

    # 3. The two heads aren't accidentally tied to each other. Arousal
    #    and valence labels differ on ~50% of samples by construction, so
    #    a model learning independent heads will produce different
    #    predictions on those samples. If arousal_pred == valence_pred
    #    elementwise, the heads have collapsed into one.
    disagreements = int(np.sum(a_pred != v_pred))
    assert disagreements > 0, "heads produced identical predictions — probably tied"
