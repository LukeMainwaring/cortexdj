"""Training script for EEG emotion classifiers.

Supports training the custom EEGNet (on DE features) or fine-tuning
CBraMod pretrained model (on raw EEG), with LOSO or grouped cross-validation.

Usage:
    uv run train-model                       # best-effort defaults (CBraMod, LOSO, 50 epochs)
    uv run train-model --quick               # fast dev run (10 epochs, 3 folds)
    uv run train-model --model eegnet        # train EEGNet instead
    uv run compare-models                    # compare from checkpoints
    uv run compare-models --retrain --quick  # retrain both quickly
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR
from torch.utils.data import DataLoader, Subset

from cortexdj.core.paths import CHECKPOINTS_DIR
from cortexdj.ml.dataset import (
    DEAPFeatureDataset,
    DEAPRawDataset,
    LabelSplitStrategy,
    load_dataset,
)
from cortexdj.ml.metrics import (
    MajorityBaselinePredictor,
    balanced_accuracy,
    class_weights_from_labels,
    macro_f1,
    per_class_recall,
    prediction_counts,
)
from cortexdj.ml.model import EEGNetClassifier
from cortexdj.ml.predict import EEGModel
from cortexdj.ml.pretrained import PretrainedDualHead, load_pretrained_dual_head

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Default hyperparameters ──────────────────────────────────────────────────
# These represent best-effort quality defaults. Use --quick for fast iteration.
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3
DEFAULT_FINETUNE_LR = 1e-5
# Batch size is resolved per-device at runtime (see _default_batch_size_for).
# CUDA default is tuned for A10G @ 24GB with bf16 AMP; MPS/CPU stay conservative
# so local M-series unified memory doesn't blow up.
DEFAULT_BATCH_SIZE_CUDA = 128
DEFAULT_BATCH_SIZE_MPS = 64
DEFAULT_BATCH_SIZE_CPU = 64
DEFAULT_N_FOLDS = 5
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_PATIENCE = 10
DEFAULT_SEED = 42
# Small label smoothing composes with class weights as an extra regularizer
# against confident-majority-class collapse. 0.05 is conservative.
DEFAULT_LABEL_SMOOTHING = 0.05
# Checkpoint schema version — bump when the saved metrics/config layout changes.
# v1: pre-collapse-fix checkpoints (unweighted CE, `avg_*_acc` metric keys only).
# v2: post-fix checkpoints with class weights, macro-F1, per-class recall. The
#     reader refuses to interpret v1's metrics because the `avg_*_acc` numbers
#     are majority-class collapse and silently mixing them into a comparison
#     table is the exact bait we want to avoid.
CHECKPOINT_SCHEMA_VERSION = 2

# Fine-tune phase uses stronger weight decay (standard for pretrained models)
FINETUNE_WEIGHT_DECAY_MULTIPLIER = 100
# LR warmup epochs for pretrained model phases
WARMUP_EPOCHS = 3

# Quick mode overrides for fast development
QUICK_EPOCHS = 10
QUICK_MAX_FOLDS = 3

# Canonical list of model types the `compare()` loop iterates. Typed as a
# tuple of literals so assignments to `TrainingConfig.model_type` type-check
# without a `# type: ignore[arg-type]`.
_MODEL_TYPES: tuple[Literal["eegnet", "cbramod"], ...] = ("eegnet", "cbramod")


@dataclass
class TrainingConfig:
    """Bundle of all training hyperparameters."""

    model_type: Literal["eegnet", "cbramod"] = "cbramod"
    cv_mode: Literal["loso", "grouped"] = "loso"
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    finetune_lr: float = DEFAULT_FINETUNE_LR
    # None means "auto-pick based on detected device" (resolved in train()).
    batch_size: int | None = None
    n_folds: int = DEFAULT_N_FOLDS
    max_folds: int | None = None
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    patience: int = DEFAULT_PATIENCE
    seed: int | None = DEFAULT_SEED
    no_early_stop: bool = False
    label_smoothing: float = DEFAULT_LABEL_SMOOTHING
    # Default to per-subject median split: DEAP's 1-9 Likert self-reports
    # vary a lot in per-subject scale, so thresholding at each subject's
    # own median both removes rating-scale bias and gives roughly balanced
    # labels per fold. The historical `fixed_5` threshold produced a 25/75
    # skew that trained models to collapse onto the majority class — still
    # available as an explicit opt-in for reproducing DEAP papers that
    # used the fixed > 5 threshold.
    label_split_strategy: LabelSplitStrategy = "median_per_subject"


class EarlyStopping:
    """Monitors validation metric and signals when to stop training."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_score = 0.0
        self.counter = 0

    def step(self, score: float) -> bool:
        """Returns True if training should stop."""
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def _get_device() -> torch.device:
    """Select the best available device: CUDA → MPS (Apple Silicon) → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _default_batch_size_for(device: torch.device) -> int:
    if device.type == "cuda":
        return DEFAULT_BATCH_SIZE_CUDA
    if device.type == "mps":
        return DEFAULT_BATCH_SIZE_MPS
    return DEFAULT_BATCH_SIZE_CPU


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Type for any dataset that has participant_ids
type ParticipantDataset = DEAPFeatureDataset | DEAPRawDataset


def make_loso_splits(
    dataset: ParticipantDataset,
    max_folds: int | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Create Leave-One-Subject-Out cross-validation splits.

    Each fold holds out all samples from one participant.
    """
    unique_ids = sorted(set(dataset.participant_ids))
    if max_folds is not None and max_folds < len(unique_ids):
        unique_ids = unique_ids[:max_folds]

    splits: list[tuple[list[int], list[int]]] = []
    for held_out_id in unique_ids:
        val_indices = [i for i, pid in enumerate(dataset.participant_ids) if pid == held_out_id]
        train_indices = [i for i, pid in enumerate(dataset.participant_ids) if pid != held_out_id]
        splits.append((train_indices, val_indices))
    return splits


def make_grouped_splits(
    dataset: ParticipantDataset,
    n_folds: int = 5,
) -> list[tuple[list[int], list[int]]]:
    """Create participant-grouped K-fold splits (no data leakage)."""
    unique_ids = sorted(set(dataset.participant_ids))
    if n_folds > len(unique_ids):
        msg = f"n_folds ({n_folds}) exceeds number of participants ({len(unique_ids)})"
        raise ValueError(msg)
    fold_size = len(unique_ids) // n_folds

    splits: list[tuple[list[int], list[int]]] = []
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(unique_ids)
        val_participant_ids = set(unique_ids[start:end])

        val_indices = [i for i, pid in enumerate(dataset.participant_ids) if pid in val_participant_ids]
        train_indices = [i for i, pid in enumerate(dataset.participant_ids) if pid not in val_participant_ids]
        splits.append((train_indices, val_indices))
    return splits


@torch.inference_mode()
def _evaluate(
    model: EEGModel,
    loader: DataLoader[tuple[torch.Tensor, int, int]],
    device: torch.device,
) -> dict[str, object]:
    """Evaluate on a val loader, returning accuracy + balanced-acc + macro-F1
    + per-class recall + prediction counts for both heads.

    Returns an `object`-valued dict because the per-class recall and
    prediction-count entries are lists; individual fold metrics and the
    average-across-folds aggregation handle both scalar and list values.
    """
    model.eval()
    non_blocking = device.type == "cuda"
    arousal_true: list[int] = []
    arousal_pred: list[int] = []
    valence_true: list[int] = []
    valence_pred: list[int] = []

    for features, arousal_labels, valence_labels in loader:
        features = features.to(device, non_blocking=non_blocking)
        arousal_targets = arousal_labels.to(device, non_blocking=non_blocking)
        valence_targets = valence_labels.to(device, non_blocking=non_blocking)

        arousal_logits, valence_logits = model(features)
        arousal_true.extend(arousal_targets.cpu().tolist())
        arousal_pred.extend(arousal_logits.argmax(1).cpu().tolist())
        valence_true.extend(valence_targets.cpu().tolist())
        valence_pred.extend(valence_logits.argmax(1).cpu().tolist())

    a_true = np.asarray(arousal_true, dtype=np.int64)
    a_pred = np.asarray(arousal_pred, dtype=np.int64)
    v_true = np.asarray(valence_true, dtype=np.int64)
    v_pred = np.asarray(valence_pred, dtype=np.int64)

    total = int(a_true.size)
    arousal_acc = float((a_pred == a_true).sum()) / total if total else 0.0
    valence_acc = float((v_pred == v_true).sum()) / total if total else 0.0
    arousal_bal = balanced_accuracy(a_true, a_pred)
    valence_bal = balanced_accuracy(v_true, v_pred)
    arousal_f1 = macro_f1(a_true, a_pred)
    valence_f1 = macro_f1(v_true, v_pred)

    return {
        "arousal_acc": arousal_acc,
        "valence_acc": valence_acc,
        "avg_acc": (arousal_acc + valence_acc) / 2,
        "arousal_balanced_acc": arousal_bal,
        "valence_balanced_acc": valence_bal,
        "avg_balanced_acc": (arousal_bal + valence_bal) / 2,
        "arousal_macro_f1": arousal_f1,
        "valence_macro_f1": valence_f1,
        "avg_macro_f1": (arousal_f1 + valence_f1) / 2,
        "arousal_recall": per_class_recall(a_true, a_pred),
        "valence_recall": per_class_recall(v_true, v_pred),
        "arousal_pred_counts": prediction_counts(a_pred),
        "valence_pred_counts": prediction_counts(v_pred),
    }


def _metric_float(metrics: dict[str, object], key: str) -> float:
    """Narrow a scalar entry in the `_evaluate` dict to float for comparisons.

    Raises `TypeError` instead of asserting so the contract is enforced
    even under `python -O`, where `assert` statements are compiled out.
    """
    value = metrics[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"expected scalar for metric {key!r}, got {type(value).__name__}")
    return float(value)


def _build_fold_criteria(
    dataset: ParticipantDataset,
    train_indices: list[int],
    device: torch.device,
    label_smoothing: float,
) -> tuple[nn.CrossEntropyLoss, nn.CrossEntropyLoss]:
    """Per-head class-weighted + label-smoothed CE criteria for a fold.

    Reads labels via `dataset.get_labels(train_indices)` so the fold
    function has no DataLoader dependency and can be unit-tested against
    any dataset that implements the same interface. Arousal and valence
    get independent weights because their class distributions differ
    (under `fixed_5` DEAP is 24.6% high arousal vs 22.3% high valence;
    under `median_per_subject` both axes are roughly balanced).
    """
    train_arousal, train_valence = dataset.get_labels(train_indices)
    arousal_weights = class_weights_from_labels(train_arousal).to(device)
    valence_weights = class_weights_from_labels(train_valence).to(device)
    logger.info(
        "  Class weights — arousal: %s, valence: %s",
        [round(w, 3) for w in arousal_weights.tolist()],
        [round(w, 3) for w in valence_weights.tolist()],
    )
    arousal_criterion = nn.CrossEntropyLoss(weight=arousal_weights, label_smoothing=label_smoothing)
    valence_criterion = nn.CrossEntropyLoss(weight=valence_weights, label_smoothing=label_smoothing)
    return arousal_criterion, valence_criterion


def _log_fold_epoch_metrics(
    prefix: str,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_total: int,
    val_metrics: dict[str, object],
    elapsed: float | None = None,
) -> None:
    """Single shared log line for per-epoch progress across both train-fold funcs."""
    arousal_acc = _metric_float(val_metrics, "arousal_acc")
    valence_acc = _metric_float(val_metrics, "valence_acc")
    avg_f1 = _metric_float(val_metrics, "avg_macro_f1")
    arousal_counts = val_metrics["arousal_pred_counts"]
    valence_counts = val_metrics["valence_pred_counts"]
    extras = f" | {elapsed:.1f}s" if elapsed is not None else ""
    logger.info(
        "  %s Epoch %d/%d | Loss: %.4f | Val acc A/V: %.3f/%.3f | macro-F1: %.3f | pred A %s V %s%s",
        prefix,
        epoch,
        total_epochs,
        train_loss / train_total if train_total else 0.0,
        arousal_acc,
        valence_acc,
        avg_f1,
        arousal_counts,
        valence_counts,
        extras,
    )


def train_fold_eegnet(
    train_loader: DataLoader[tuple[torch.Tensor, int, int]],
    val_loader: DataLoader[tuple[torch.Tensor, int, int]],
    *,
    arousal_criterion: nn.CrossEntropyLoss,
    valence_criterion: nn.CrossEntropyLoss,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[EEGNetClassifier, dict[str, object]]:
    """Train EEGNet on DE features for one fold.

    Criteria are built by the caller so this function has no dependency
    on the dataset — it only needs the loaders, the loss objects, and
    the config.
    """
    model = EEGNetClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    stopper = EarlyStopping(patience=config.patience)
    use_amp = device.type == "cuda"
    # non_blocking=True is only meaningful with pinned CUDA memory. On
    # MPS/CPU there's no pinned allocator to pipeline against, and PyTorch
    # 2.9-2.11 has had intermittent MPS async-correctness regressions on
    # that path — gate the flag on CUDA to match the environment it's
    # designed for.
    non_blocking = device.type == "cuda"

    best_val_f1 = -1.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_total = 0
        nan_skipped = 0

        for features, arousal_labels, valence_labels in train_loader:
            features = features.to(device, non_blocking=non_blocking)
            arousal_targets = arousal_labels.to(device, non_blocking=non_blocking)
            valence_targets = valence_labels.to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    arousal_logits, valence_logits = model(features)
                    loss = arousal_criterion(arousal_logits, arousal_targets) + valence_criterion(
                        valence_logits, valence_targets
                    )
            else:
                arousal_logits, valence_logits = model(features)
                loss = arousal_criterion(arousal_logits, arousal_targets) + valence_criterion(
                    valence_logits, valence_targets
                )

            if not torch.isfinite(loss):
                nan_skipped += 1
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_total += features.size(0)

        if nan_skipped:
            logger.warning(
                "  Epoch %d/%d: skipped %d non-finite-loss batches",
                epoch + 1,
                config.epochs,
                nan_skipped,
            )
        scheduler.step()
        val_metrics = _evaluate(model, val_loader, device)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            _log_fold_epoch_metrics("", epoch + 1, config.epochs, train_loss, train_total, val_metrics)

        val_f1 = _metric_float(val_metrics, "avg_macro_f1")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if not config.no_early_stop and stopper.step(val_f1):
            logger.info(f"  Early stopping at epoch {epoch + 1} (patience={config.patience})")
            break

    model.load_state_dict(best_state)
    return model, _evaluate(model, val_loader, device)


def train_fold_pretrained(
    train_loader: DataLoader[tuple[torch.Tensor, int, int]],
    val_loader: DataLoader[tuple[torch.Tensor, int, int]],
    *,
    arousal_criterion: nn.CrossEntropyLoss,
    valence_criterion: nn.CrossEntropyLoss,
    base_model: PretrainedDualHead,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[PretrainedDualHead, dict[str, object]]:
    """Train pretrained model for one fold with two-phase strategy.

    Phase 1 (first 1/3 epochs): frozen backbone, train heads only with warmup.
    Phase 2 (remaining epochs): unfreeze backbone, full fine-tuning with warmup + cosine decay.

    Reinitializes heads from scratch each fold but reuses the pretrained backbone
    weights (avoids re-downloading). Class-weighted criteria are supplied by the
    caller (see `_build_fold_criteria`).
    """
    # Deep copy so each fold starts fresh, reusing cached backbone weights
    model = copy.deepcopy(base_model).to(device)
    model.freeze_backbone()
    use_amp = device.type == "cuda"
    non_blocking = device.type == "cuda"

    phase1_epochs = max(1, config.epochs // 3)
    phase2_epochs = config.epochs - phase1_epochs

    best_val_f1 = -1.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Phase 1: frozen backbone — train heads with warmup
    head_params = list(model.arousal_head.parameters()) + list(model.valence_head.parameters())
    optimizer = torch.optim.AdamW(head_params, lr=config.lr, weight_decay=config.weight_decay)
    warmup_iters = min(WARMUP_EPOCHS, phase1_epochs)
    scheduler: LRScheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
    phase1_patience = max(3, config.patience // 2)
    stopper = EarlyStopping(patience=phase1_patience)

    for epoch in range(phase1_epochs):
        epoch_start = time.monotonic()
        model.train()
        train_loss = 0.0
        train_total = 0
        nan_skipped = 0

        for features, arousal_labels, valence_labels in train_loader:
            features = features.to(device, non_blocking=non_blocking)
            arousal_targets = arousal_labels.to(device, non_blocking=non_blocking)
            valence_targets = valence_labels.to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    arousal_logits, valence_logits = model(features)
                    loss = arousal_criterion(arousal_logits, arousal_targets) + valence_criterion(
                        valence_logits, valence_targets
                    )
            else:
                arousal_logits, valence_logits = model(features)
                loss = arousal_criterion(arousal_logits, arousal_targets) + valence_criterion(
                    valence_logits, valence_targets
                )

            if not torch.isfinite(loss):
                nan_skipped += 1
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_total += features.size(0)

        if nan_skipped:
            logger.warning(
                "  [Phase 1] Epoch %d/%d: skipped %d non-finite-loss batches",
                epoch + 1,
                phase1_epochs,
                nan_skipped,
            )

        scheduler.step()
        val_metrics = _evaluate(model, val_loader, device)
        elapsed = time.monotonic() - epoch_start
        _log_fold_epoch_metrics("[Phase 1]", epoch + 1, phase1_epochs, train_loss, train_total, val_metrics, elapsed)

        val_f1 = _metric_float(val_metrics, "avg_macro_f1")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if not config.no_early_stop and stopper.step(val_f1):
            logger.info(f"  Phase 1 early stop at epoch {epoch + 1} (patience={phase1_patience})")
            break

    # Phase 2: full fine-tuning — warmup + cosine decay
    # Restore best Phase 1 weights so we don't start from a degraded state
    model.load_state_dict(best_state)
    model.unfreeze_backbone()
    finetune_wd = config.weight_decay * FINETUNE_WEIGHT_DECAY_MULTIPLIER
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.finetune_lr, weight_decay=finetune_wd)
    warmup_iters = min(WARMUP_EPOCHS, phase2_epochs)
    cosine_epochs = max(1, phase2_epochs - warmup_iters)
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    scheduler = SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_iters])
    phase2_patience = max(5, config.patience)
    stopper = EarlyStopping(patience=phase2_patience)

    for epoch in range(phase2_epochs):
        epoch_start = time.monotonic()
        model.train()
        train_loss = 0.0
        train_total = 0
        nan_skipped = 0

        for features, arousal_labels, valence_labels in train_loader:
            features = features.to(device, non_blocking=non_blocking)
            arousal_targets = arousal_labels.to(device, non_blocking=non_blocking)
            valence_targets = valence_labels.to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    arousal_logits, valence_logits = model(features)
                    loss = arousal_criterion(arousal_logits, arousal_targets) + valence_criterion(
                        valence_logits, valence_targets
                    )
            else:
                arousal_logits, valence_logits = model(features)
                loss = arousal_criterion(arousal_logits, arousal_targets) + valence_criterion(
                    valence_logits, valence_targets
                )

            if not torch.isfinite(loss):
                nan_skipped += 1
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_total += features.size(0)

        if nan_skipped:
            logger.warning(
                "  [Phase 2] Epoch %d/%d: skipped %d non-finite-loss batches",
                epoch + 1,
                phase2_epochs,
                nan_skipped,
            )

        scheduler.step()
        val_metrics = _evaluate(model, val_loader, device)
        elapsed = time.monotonic() - epoch_start

        if (epoch + 1) % 5 == 0 or epoch == 0:
            _log_fold_epoch_metrics(
                "[Phase 2]", epoch + 1, phase2_epochs, train_loss, train_total, val_metrics, elapsed
            )

        val_f1 = _metric_float(val_metrics, "avg_macro_f1")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if not config.no_early_stop and stopper.step(val_f1):
            logger.info(f"  Phase 2 early stop at epoch {epoch + 1} (patience={phase2_patience})")
            break

    model.load_state_dict(best_state)
    return model, _evaluate(model, val_loader, device)


def _dataloader_kwargs(device: torch.device) -> dict[str, object]:
    """DataLoader kwargs optimized per device.

    macOS/MPS hangs with num_workers > 0 due to fork-related deadlocks,
    so we only parallelize data loading on CUDA. On Modal A10G (8 vCPUs),
    8 workers + prefetch_factor=4 keeps the GPU fed once bf16 AMP shortens
    step time.
    """
    if device.type == "cuda":
        n_workers = min(8, os.cpu_count() or 1)
        return {
            "num_workers": n_workers,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
        }
    return {"num_workers": 0, "pin_memory": False}


def train(config: TrainingConfig) -> None:
    if config.seed is not None:
        _set_seed(config.seed)

    device = _get_device()

    # CUDA-only backend optimizations. `set_float32_matmul_precision("high")`
    # enables TF32 on CUDA matmuls and is a no-op on MPS/CPU — gated for
    # clarity. `cudnn.benchmark` is safe for LOSO: train/val loaders feed
    # near-fixed shapes (batch × 32 × 800 for CBraMod, batch × 160 for
    # EEGNet). The tail batch is smaller so cudnn autotunes twice and
    # caches both — still ~10-15% faster than the default heuristic.
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    # Resolve into a local so we don't mutate the caller's config dataclass.
    batch_size = config.batch_size if config.batch_size is not None else _default_batch_size_for(device)
    if config.batch_size is None:
        logger.info(f"Auto-selected batch_size={batch_size} for device={device.type}")

    logger.info(f"Using device: {device}")
    logger.info(
        f"Config: model={config.model_type}, cv={config.cv_mode}, "
        f"epochs={config.epochs}, lr={config.lr}, batch_size={batch_size}"
    )

    mode: Literal["features", "raw"] = "raw" if config.model_type == "cbramod" else "features"

    try:
        dataset = load_dataset(mode=mode, label_split_strategy=config.label_split_strategy)
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    logger.info(f"Label split strategy: {config.label_split_strategy}")

    logger.info(f"Loaded {len(dataset)} segments ({mode} mode)")

    if len(dataset) == 0:
        logger.error("No data found.")
        return

    # Build CV splits
    if config.cv_mode == "loso":
        splits = make_loso_splits(dataset, max_folds=config.max_folds)
    else:
        splits = make_grouped_splits(dataset, n_folds=config.n_folds)

    logger.info(f"Cross-validation: {len(splits)} folds ({config.cv_mode})")

    # Pre-load pretrained model once (avoids re-downloading per fold)
    base_pretrained: PretrainedDualHead | None = None
    if config.model_type == "cbramod":
        logger.info("Loading pretrained CBraMod backbone (one-time)...")
        load_start = time.monotonic()
        base_pretrained = load_pretrained_dual_head()
        logger.info(f"Pretrained model loaded in {time.monotonic() - load_start:.1f}s")

    all_metrics: list[dict[str, object]] = []
    best_overall_f1 = -1.0
    best_model_state: dict[str, torch.Tensor] | None = None
    dl_kwargs = _dataloader_kwargs(device)
    train_start = time.monotonic()

    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        # Fresh RNG state per fold so we get honest per-fold variance rather
        # than each fold inheriting the previous fold's PyTorch RNG position.
        if config.seed is not None:
            _set_seed(config.seed + fold_idx)

        logger.info(f"\n--- Fold {fold_idx + 1}/{len(splits)} (train={len(train_indices)}, val={len(val_indices)}) ---")

        # Build per-fold criteria from training indices before creating the
        # DataLoader — reads labels from `dataset.samples` directly, so we
        # never iterate the loader twice in one fold.
        arousal_criterion, valence_criterion = _build_fold_criteria(
            dataset, train_indices, device, config.label_smoothing
        )

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        train_loader: DataLoader[tuple[torch.Tensor, int, int]] = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            **dl_kwargs,  # type: ignore[arg-type]
        )
        val_loader: DataLoader[tuple[torch.Tensor, int, int]] = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            **dl_kwargs,  # type: ignore[arg-type]
        )

        fold_model: EEGModel
        if config.model_type == "cbramod":
            assert base_pretrained is not None
            fold_model, metrics = train_fold_pretrained(
                train_loader,
                val_loader,
                arousal_criterion=arousal_criterion,
                valence_criterion=valence_criterion,
                base_model=base_pretrained,
                config=config,
                device=device,
            )
        else:
            fold_model, metrics = train_fold_eegnet(
                train_loader,
                val_loader,
                arousal_criterion=arousal_criterion,
                valence_criterion=valence_criterion,
                config=config,
                device=device,
            )

        all_metrics.append(metrics)

        # Track best fold for checkpoint (macro-F1, not raw accuracy —
        # accuracy is dominated by the class prior on skewed DEAP labels
        # and was exactly what made the collapse invisible).
        fold_f1 = _metric_float(metrics, "avg_macro_f1")
        if fold_f1 > best_overall_f1:
            best_overall_f1 = fold_f1
            best_model_state = {k: v.cpu().clone() for k, v in fold_model.state_dict().items()}

        # Drop GPU memory before the next fold. Over 32 LOSO folds, the fold's
        # model (deepcopied CBraMod backbone + optimizer state + activations)
        # would otherwise accumulate and fragment the allocator.
        del fold_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_time = time.monotonic() - train_start
    _print_results_table(all_metrics, cv_mode=config.cv_mode, total_time=total_time)

    # Save best fold's model
    if best_model_state is not None:
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_name = "cbramod_best.pt" if config.model_type == "cbramod" else "eegnet_best.pt"
        checkpoint_path = CHECKPOINTS_DIR / checkpoint_name
        torch.save(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "model_type": config.model_type,
                "model_state_dict": best_model_state,
                "metrics": _aggregate_metrics(
                    all_metrics,
                    cv_mode=config.cv_mode,
                    epochs=config.epochs,
                    best_fold_f1=best_overall_f1,
                ),
                "config": {
                    "model_type": config.model_type,
                    "cv_mode": config.cv_mode,
                    "epochs": config.epochs,
                    "lr": config.lr,
                    "finetune_lr": config.finetune_lr,
                    "batch_size": batch_size,
                    "weight_decay": config.weight_decay,
                    "max_grad_norm": config.max_grad_norm,
                    "patience": config.patience,
                    "seed": config.seed,
                    "label_smoothing": config.label_smoothing,
                    "label_split_strategy": config.label_split_strategy,
                },
                "fold_metrics": all_metrics,
                "training_time": total_time,
            },
            checkpoint_path,
        )
        logger.info(f"Saved best model to {checkpoint_path}")


def _std(values: list[float]) -> float:
    """Sample standard deviation (Bessel-corrected)."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


_SCALAR_METRIC_KEYS: tuple[str, ...] = (
    "arousal_acc",
    "valence_acc",
    "avg_acc",
    "arousal_balanced_acc",
    "valence_balanced_acc",
    "avg_balanced_acc",
    "arousal_macro_f1",
    "valence_macro_f1",
    "avg_macro_f1",
)


def _aggregate_metrics(
    fold_metrics: list[dict[str, object]],
    *,
    cv_mode: str,
    epochs: int,
    best_fold_f1: float,
) -> dict[str, object]:
    """Average scalar metrics across folds for the checkpoint `metrics` entry."""
    n = len(fold_metrics)
    if n == 0:
        return {}

    averaged: dict[str, object] = {}
    for key in _SCALAR_METRIC_KEYS:
        averaged[key] = sum(_metric_float(m, key) for m in fold_metrics) / n

    averaged["best_fold_macro_f1"] = best_fold_f1
    averaged["n_folds"] = n
    averaged["cv_mode"] = cv_mode
    averaged["epochs"] = epochs
    return averaged


def _print_results_table(
    all_metrics: list[dict[str, object]],
    *,
    cv_mode: str,
    total_time: float,
) -> None:
    """Print per-fold results with mean ± std and wall-clock time.

    Accuracy is shown so past checkpoint numbers remain comparable, but
    macro-F1 is the honest headline metric on DEAP's skewed labels.
    """
    n = len(all_metrics)
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Cross-Validation Results ({n} folds, {cv_mode})")
    logger.info(f"{'=' * 80}")
    logger.info(
        f"{'Fold':<6} | {'A Acc':>7} | {'V Acc':>7} | {'Avg Acc':>8} | {'A F1':>6} | {'V F1':>6} | {'Avg F1':>7}"
    )
    logger.info(f"{'-' * 6}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 8}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 7}")

    for i, m in enumerate(all_metrics):
        logger.info(
            f"{i + 1:<6} | {_metric_float(m, 'arousal_acc'):>7.4f} | "
            f"{_metric_float(m, 'valence_acc'):>7.4f} | "
            f"{_metric_float(m, 'avg_acc'):>8.4f} | "
            f"{_metric_float(m, 'arousal_macro_f1'):>6.3f} | "
            f"{_metric_float(m, 'valence_macro_f1'):>6.3f} | "
            f"{_metric_float(m, 'avg_macro_f1'):>7.3f}"
        )

    def _mean(key: str) -> float:
        return sum(_metric_float(m, key) for m in all_metrics) / n

    def _std_key(key: str) -> float:
        return _std([_metric_float(m, key) for m in all_metrics])

    logger.info(f"{'-' * 6}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 8}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 7}")
    logger.info(
        f"{'Mean':<6} | {_mean('arousal_acc'):>7.4f} | {_mean('valence_acc'):>7.4f} | "
        f"{_mean('avg_acc'):>8.4f} | {_mean('arousal_macro_f1'):>6.3f} | "
        f"{_mean('valence_macro_f1'):>6.3f} | {_mean('avg_macro_f1'):>7.3f}"
    )
    logger.info(
        f"{'Std':<6} | {_std_key('arousal_acc'):>7.4f} | {_std_key('valence_acc'):>7.4f} | "
        f"{_std_key('avg_acc'):>8.4f} | {_std_key('arousal_macro_f1'):>6.3f} | "
        f"{_std_key('valence_macro_f1'):>6.3f} | {_std_key('avg_macro_f1'):>7.3f}"
    )

    # Per-class recall summary — the fastest way to eyeball collapse.
    def _mean_list(key: str, idx: int) -> float:
        values: list[float] = []
        for m in all_metrics:
            entry = m[key]
            if not isinstance(entry, list):
                raise TypeError(f"expected list for metric {key!r}, got {type(entry).__name__}")
            values.append(float(entry[idx]))
        return sum(values) / n

    arousal_low = _mean_list("arousal_recall", 0)
    arousal_high = _mean_list("arousal_recall", 1)
    valence_low = _mean_list("valence_recall", 0)
    valence_high = _mean_list("valence_recall", 1)
    logger.info(
        f"\nMean recall — arousal: low={arousal_low:.3f} high={arousal_high:.3f} | "
        f"valence: low={valence_low:.3f} high={valence_high:.3f}"
    )

    minutes, seconds = divmod(total_time, 60)
    logger.info(f"\nTotal training time: {int(minutes)}m {seconds:.1f}s")


def _print_comparison_table(results: dict[str, dict[str, float]], source: str) -> None:
    logger.info(f"\n{'=' * 76}")
    logger.info(f"Model Comparison on {source.upper()}")
    logger.info(f"{'=' * 76}")
    logger.info(
        f"{'Model':<18} | {'A Acc':>6} | {'V Acc':>6} | {'Avg Acc':>7} | {'A F1':>6} | {'V F1':>6} | {'Avg F1':>7}"
    )
    logger.info(f"{'-' * 18}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 7}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 7}")
    labels = {
        "majority": "MajorityBaseline",
        "eegnet": "EEGNet (DE)",
        "cbramod": "CBraMod (FT)",
    }
    for name, metrics in results.items():
        label = labels.get(name, name)
        logger.info(
            f"{label:<18} | {metrics.get('arousal_acc', 0.0):>6.3f} | "
            f"{metrics.get('valence_acc', 0.0):>6.3f} | "
            f"{metrics.get('avg_acc', 0.0):>7.3f} | "
            f"{metrics.get('arousal_macro_f1', 0.0):>6.3f} | "
            f"{metrics.get('valence_macro_f1', 0.0):>6.3f} | "
            f"{metrics.get('avg_macro_f1', 0.0):>7.3f}"
        )


def _majority_baseline_metrics(dataset: ParticipantDataset, config: TrainingConfig) -> dict[str, float]:
    """Compute per-fold MajorityBaseline metrics, averaged across folds.

    For each fold: fit MajorityBaselinePredictor on the training arousal
    and valence labels, predict constant on val, score with the same
    metrics the trained models use. Provides the reference point that
    makes collapse visually obvious in the comparison table.
    """
    if config.cv_mode == "loso":
        splits = make_loso_splits(dataset, max_folds=config.max_folds)
    else:
        splits = make_grouped_splits(dataset, n_folds=config.n_folds)

    arousal_labels, valence_labels = dataset.get_labels()

    accum: dict[str, float] = {k: 0.0 for k in _SCALAR_METRIC_KEYS}
    for train_idx, val_idx in splits:
        arousal_baseline = MajorityBaselinePredictor()
        arousal_baseline.fit(arousal_labels[train_idx])
        valence_baseline = MajorityBaselinePredictor()
        valence_baseline.fit(valence_labels[train_idx])

        a_true = arousal_labels[val_idx]
        v_true = valence_labels[val_idx]
        a_pred = arousal_baseline.predict(len(val_idx))
        v_pred = valence_baseline.predict(len(val_idx))

        a_acc = float((a_pred == a_true).mean()) if len(val_idx) else 0.0
        v_acc = float((v_pred == v_true).mean()) if len(val_idx) else 0.0
        a_bal = balanced_accuracy(a_true, a_pred)
        v_bal = balanced_accuracy(v_true, v_pred)
        a_f1 = macro_f1(a_true, a_pred)
        v_f1 = macro_f1(v_true, v_pred)

        accum["arousal_acc"] += a_acc
        accum["valence_acc"] += v_acc
        accum["avg_acc"] += (a_acc + v_acc) / 2
        accum["arousal_balanced_acc"] += a_bal
        accum["valence_balanced_acc"] += v_bal
        accum["avg_balanced_acc"] += (a_bal + v_bal) / 2
        accum["arousal_macro_f1"] += a_f1
        accum["valence_macro_f1"] += v_f1
        accum["avg_macro_f1"] += (a_f1 + v_f1) / 2

    n = len(splits)
    if n == 0:
        return accum
    return {k: v / n for k, v in accum.items()}


_STALE_CHECKPOINT_ROW: dict[str, float] = {
    "arousal_acc": 0.0,
    "valence_acc": 0.0,
    "avg_acc": 0.0,
    "arousal_macro_f1": 0.0,
    "valence_macro_f1": 0.0,
    "avg_macro_f1": 0.0,
}


def _is_stale_checkpoint(checkpoint: dict[str, object]) -> bool:
    """True if the checkpoint predates `CHECKPOINT_SCHEMA_VERSION`.

    Treats a missing, non-int, or below-cutoff `schema_version` as stale
    — the three shapes a pre-fix or corrupted checkpoint can take. Used
    by both `_checkpoint_comparison_row` (returns zeros) and the
    `compare()` caller (logs a "retrain recommended" warning) so the
    two sites can't drift out of sync.
    """
    schema = checkpoint.get("schema_version")
    return not isinstance(schema, int) or schema < CHECKPOINT_SCHEMA_VERSION


def _checkpoint_comparison_row(checkpoint: dict[str, object]) -> dict[str, float]:
    """Pull the compare-table scalar columns out of a loaded checkpoint.

    Refuses pre-fix (`_is_stale_checkpoint`) checkpoints: returns zeros
    and lets the caller log a "retrain recommended" warning. Pre-fix
    checkpoints had `avg_*_acc` numbers dominated by majority-class
    collapse, and silently mixing them into the comparison table is the
    exact bait we want to avoid.
    """
    if _is_stale_checkpoint(checkpoint):
        return dict(_STALE_CHECKPOINT_ROW)

    metrics = checkpoint.get("metrics", {})
    if not isinstance(metrics, dict):
        return dict(_STALE_CHECKPOINT_ROW)

    def _scalar(key: str) -> float:
        value = metrics.get(key, 0.0)
        return float(value) if isinstance(value, (int, float)) else 0.0

    return {
        "arousal_acc": _scalar("arousal_acc"),
        "valence_acc": _scalar("valence_acc"),
        "avg_acc": _scalar("avg_acc"),
        "arousal_macro_f1": _scalar("arousal_macro_f1"),
        "valence_macro_f1": _scalar("valence_macro_f1"),
        "avg_macro_f1": _scalar("avg_macro_f1"),
    }


def compare(config: TrainingConfig, *, retrain: bool = False) -> None:
    """Compare EEGNet and CBraMod metrics.

    By default, loads metrics from existing checkpoints. Use retrain=True
    to train both models from scratch. A MajorityBaseline row is always
    computed from the dataset labels so collapse is immediately visible.
    """
    if not retrain:
        # Try loading metrics from existing checkpoints
        results: dict[str, dict[str, float]] = {}

        # Baseline is cheap — always compute it for the comparison table.
        try:
            # Use EEGNet's feature dataset for label distribution (labels are
            # mode-independent, but the feature dataset loads faster).
            baseline_dataset = load_dataset(mode="features", label_split_strategy=config.label_split_strategy)
            results["majority"] = _majority_baseline_metrics(baseline_dataset, config)
            logger.info("Computed MajorityBaseline from dataset labels")
        except FileNotFoundError as e:
            logger.warning(f"Could not compute MajorityBaseline: {e}")

        checkpoint_map = {
            "eegnet": CHECKPOINTS_DIR / "eegnet_best.pt",
            "cbramod": CHECKPOINTS_DIR / "cbramod_best.pt",
        }

        for model_type, path in checkpoint_map.items():
            if path.exists():
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                results[model_type] = _checkpoint_comparison_row(checkpoint)
                m = checkpoint.get("metrics", {})
                if isinstance(m, dict):
                    info = f"cv={m.get('cv_mode', '?')}, {m.get('n_folds', '?')} folds, {m.get('epochs', '?')} epochs"
                else:
                    info = "metrics: <unreadable>"
                if _is_stale_checkpoint(checkpoint):
                    schema = checkpoint.get("schema_version", "<missing>")
                    logger.warning(
                        "Loaded %s checkpoint (%s) — pre-fix schema %r, retrain recommended. "
                        "Comparison row will show zeros.",
                        model_type,
                        info,
                        schema,
                    )
                else:
                    logger.info(f"Loaded {model_type} checkpoint ({info})")
            else:
                logger.warning(f"No checkpoint found for {model_type} at {path}")

        trained_count = sum(1 for k in ("eegnet", "cbramod") if k in results)
        if trained_count >= 2:
            _print_comparison_table(results, "deap")
            return
        elif trained_count:
            logger.info("Only one checkpoint found. Retraining missing model(s)...\n")
        else:
            logger.info("No checkpoints found. Training both models...\n")

    # Full retraining path
    if config.seed is not None:
        _set_seed(config.seed)

    device = _get_device()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    # Resolve into a local so we don't mutate the caller's config dataclass.
    batch_size = config.batch_size if config.batch_size is not None else _default_batch_size_for(device)
    if config.batch_size is None:
        logger.info(f"Auto-selected batch_size={batch_size} for device={device.type}")

    results = {}

    for model_type in _MODEL_TYPES:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training {model_type}...")
        logger.info(f"{'=' * 60}\n")

        mode: Literal["features", "raw"] = "raw" if model_type == "cbramod" else "features"
        dataset = load_dataset(mode=mode, label_split_strategy=config.label_split_strategy)

        if config.cv_mode == "loso":
            splits = make_loso_splits(dataset, max_folds=config.max_folds)
        else:
            splits = make_grouped_splits(dataset, n_folds=config.n_folds)

        all_metrics: list[dict[str, object]] = []

        # Create a per-model config so fold functions see the right model_type
        model_config = TrainingConfig(
            model_type=model_type,
            cv_mode=config.cv_mode,
            epochs=config.epochs,
            lr=config.lr,
            finetune_lr=config.finetune_lr,
            batch_size=batch_size,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
            patience=config.patience,
            seed=config.seed,
            no_early_stop=config.no_early_stop,
            label_smoothing=config.label_smoothing,
            label_split_strategy=config.label_split_strategy,
        )

        base_pretrained: PretrainedDualHead | None = None
        if model_type == "cbramod":
            base_pretrained = load_pretrained_dual_head()

        dl_kwargs = _dataloader_kwargs(device)

        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            if config.seed is not None:
                _set_seed(config.seed + fold_idx)
            logger.info(f"--- Fold {fold_idx + 1}/{len(splits)} ---")

            arousal_criterion, valence_criterion = _build_fold_criteria(
                dataset, train_indices, device, config.label_smoothing
            )

            train_loader: DataLoader[tuple[torch.Tensor, int, int]] = DataLoader(
                Subset(dataset, train_indices),
                batch_size=batch_size,
                shuffle=True,
                **dl_kwargs,  # type: ignore[arg-type]
            )
            val_loader: DataLoader[tuple[torch.Tensor, int, int]] = DataLoader(
                Subset(dataset, val_indices),
                batch_size=batch_size,
                shuffle=False,
                **dl_kwargs,  # type: ignore[arg-type]
            )

            fold_model: EEGModel
            if model_type == "cbramod":
                assert base_pretrained is not None
                fold_model, metrics = train_fold_pretrained(
                    train_loader,
                    val_loader,
                    arousal_criterion=arousal_criterion,
                    valence_criterion=valence_criterion,
                    base_model=base_pretrained,
                    config=model_config,
                    device=device,
                )
            else:
                fold_model, metrics = train_fold_eegnet(
                    train_loader,
                    val_loader,
                    arousal_criterion=arousal_criterion,
                    valence_criterion=valence_criterion,
                    config=model_config,
                    device=device,
                )
            all_metrics.append(metrics)
            del fold_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        n_splits = len(splits)
        results[model_type] = {
            key: sum(_metric_float(m, key) for m in all_metrics) / n_splits
            for key in ("arousal_acc", "valence_acc", "avg_acc", "arousal_macro_f1", "valence_macro_f1", "avg_macro_f1")
        }

    # Also compute majority baseline for the retrain path so the table
    # always has the reference row.
    try:
        baseline_dataset = load_dataset(mode="features", label_split_strategy=config.label_split_strategy)
        results["majority"] = _majority_baseline_metrics(baseline_dataset, config)
    except FileNotFoundError as e:
        logger.warning(f"Could not compute MajorityBaseline: {e}")

    _print_comparison_table(results, "deap")


def _build_train_parser() -> argparse.ArgumentParser:
    """Build the shared argument parser for training CLI commands."""
    parser = argparse.ArgumentParser(description="Train EEG emotion classifier")
    parser.add_argument(
        "--model",
        choices=["eegnet", "cbramod"],
        default="cbramod",
        help="Model type (default: cbramod)",
    )
    parser.add_argument(
        "--cv",
        choices=["loso", "grouped"],
        default="loso",
        help="Cross-validation strategy (default: loso)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help=f"Training epochs per fold (default: {DEFAULT_EPOCHS})"
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=DEFAULT_FINETUNE_LR,
        help=f"Fine-tuning LR for pretrained backbone (default: {DEFAULT_FINETUNE_LR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(f"Batch size (default: {DEFAULT_BATCH_SIZE_CUDA} on CUDA, {DEFAULT_BATCH_SIZE_MPS} on MPS/CPU)"),
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help=f"Number of CV folds for grouped mode (default: {DEFAULT_N_FOLDS})",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Limit LOSO folds for faster development",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help=f"Weight decay for AdamW (default: {DEFAULT_WEIGHT_DECAY})",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=DEFAULT_MAX_GRAD_NORM,
        help=f"Max gradient norm for clipping (default: {DEFAULT_MAX_GRAD_NORM})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Early stopping patience in epochs (default: {DEFAULT_PATIENCE})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help=f"Random seed for reproducibility (default: {DEFAULT_SEED})"
    )
    parser.add_argument("--no-seed", action="store_true", help="Disable seeding for non-deterministic training")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping (train for all epochs)")
    parser.add_argument(
        "--label-split",
        choices=["fixed_5", "median_global", "median_per_subject"],
        default="median_per_subject",
        help=(
            "DEAP label binarization strategy. median_per_subject (default) "
            "splits at each subject's own Likert median, giving balanced "
            "labels per fold. fixed_5 uses the historical >= 5 threshold, "
            "which produces a ~25/75 skew and is only useful for reproducing "
            "papers that adopted it."
        ),
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=DEFAULT_LABEL_SMOOTHING,
        help=f"Cross-entropy label smoothing (default: {DEFAULT_LABEL_SMOOTHING})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Fast dev mode: {QUICK_EPOCHS} epochs, {QUICK_MAX_FOLDS} folds (overrides --epochs/--max-folds if not explicitly set)",
    )
    return parser


def _resolve_config(args: argparse.Namespace) -> TrainingConfig:
    """Build TrainingConfig from parsed CLI args, applying --quick overrides."""
    epochs = args.epochs
    max_folds = args.max_folds
    if args.quick:
        if epochs is None:
            epochs = QUICK_EPOCHS
        if max_folds is None:
            max_folds = QUICK_MAX_FOLDS
    if epochs is None:
        epochs = DEFAULT_EPOCHS

    return TrainingConfig(
        model_type=args.model,
        cv_mode=args.cv,
        epochs=epochs,
        lr=args.lr,
        finetune_lr=args.finetune_lr,
        batch_size=args.batch_size,
        n_folds=args.folds,
        max_folds=max_folds,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        seed=None if args.no_seed else args.seed,
        no_early_stop=args.no_early_stop,
        label_smoothing=args.label_smoothing,
        label_split_strategy=args.label_split,
    )


def main() -> None:
    parser = _build_train_parser()
    args = parser.parse_args()
    train(_resolve_config(args))


def compare_main() -> None:
    parser = _build_train_parser()
    parser.description = "Compare EEGNet vs CBraMod on DEAP"
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining instead of loading existing checkpoints",
    )
    args = parser.parse_args()
    compare(_resolve_config(args), retrain=args.retrain)
