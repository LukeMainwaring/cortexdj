"""Frozen data + evaluation surface for autoresearch.

The autoresearch agent is NOT allowed to modify this file. It fixes the
data split, the random seed, the wall-clock budget, and the evaluation
metric so that experiment results are comparable across runs regardless
of what the agent changes in ``train.py``.

If you need to change anything here (e.g. the val subject split, the
label strategy, or the metric), treat that as a step-change in the
research setup: reset the experiment log, re-run the baseline, and make
the change as a human.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from cortexdj.core.paths import DEAP_DATA_DIR
from cortexdj.ml.dataset import DEAPFeatureDataset
from cortexdj.ml.metrics import macro_f1
from cortexdj.ml.preprocessing import FREQ_BANDS

SEED = 42
WALL_CLOCK_BUDGET_SECONDS = int(os.environ.get("WALL_CLOCK_BUDGET_SECONDS", "900"))
VAL_SUBJECTS: tuple[int, ...] = (29, 30, 31, 32)
LABEL_SPLIT_STRATEGY = "median_per_subject"

NUM_CHANNELS = 32
NUM_BANDS = len(FREQ_BANDS)
FEATURE_DIM = NUM_CHANNELS * NUM_BANDS  # 160


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_splits() -> tuple[Subset[Any], Subset[Any]]:
    """Return (train, val) Subsets over DE features with a fixed subject split.

    Reuses the cached ``.npz`` produced by ``cortexdj.ml.dataset`` — so
    after the first run, this call is essentially instant. The held-out
    subjects are fixed (``VAL_SUBJECTS``); every experiment scores against
    the same 4 subjects.
    """
    full = DEAPFeatureDataset(DEAP_DATA_DIR, label_split_strategy=LABEL_SPLIT_STRATEGY)
    val_set = set(VAL_SUBJECTS)
    train_idx = [i for i, pid in enumerate(full.participant_ids) if pid not in val_set]
    val_idx = [i for i, pid in enumerate(full.participant_ids) if pid in val_set]
    if not train_idx or not val_idx:
        msg = (
            f"Empty split: train={len(train_idx)} val={len(val_idx)}. "
            f"Check VAL_SUBJECTS={VAL_SUBJECTS} against available participants."
        )
        raise RuntimeError(msg)
    return Subset(full, train_idx), Subset(full, val_idx)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_ds: Subset[Any],
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, float]:
    """Run the dual-head classifier on the val set, return macro-F1 metrics.

    Contract the agent must preserve: ``model(features)`` returns
    ``(arousal_logits, valence_logits)`` where each is shape (batch, 2).
    """
    was_training = model.training
    model.eval()
    a_trues: list[np.typing.NDArray[np.int64]] = []
    a_preds: list[np.typing.NDArray[np.int64]] = []
    v_trues: list[np.typing.NDArray[np.int64]] = []
    v_preds: list[np.typing.NDArray[np.int64]] = []
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    for features, arousal, valence in loader:
        features = features.to(device)
        a_logits, v_logits = model(features)
        a_preds.append(a_logits.argmax(dim=1).cpu().numpy().astype(np.int64))
        v_preds.append(v_logits.argmax(dim=1).cpu().numpy().astype(np.int64))
        a_trues.append(arousal.numpy().astype(np.int64))
        v_trues.append(valence.numpy().astype(np.int64))
    if was_training:
        model.train()

    a_true = np.concatenate(a_trues)
    a_pred = np.concatenate(a_preds)
    v_true = np.concatenate(v_trues)
    v_pred = np.concatenate(v_preds)
    a_f1 = macro_f1(a_true, a_pred, num_classes=2)
    v_f1 = macro_f1(v_true, v_pred, num_classes=2)
    return {
        "avg_macro_f1": (a_f1 + v_f1) / 2.0,
        "arousal_f1": a_f1,
        "valence_f1": v_f1,
    }
