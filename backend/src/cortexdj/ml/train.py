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
    load_dataset,
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

# Fine-tune phase uses stronger weight decay (standard for pretrained models)
FINETUNE_WEIGHT_DECAY_MULTIPLIER = 100
# LR warmup epochs for pretrained model phases
WARMUP_EPOCHS = 3

# Quick mode overrides for fast development
QUICK_EPOCHS = 10
QUICK_MAX_FOLDS = 3


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
) -> dict[str, float]:
    """Evaluate model on a data loader, returning accuracy metrics."""
    model.eval()
    correct_arousal = 0
    correct_valence = 0
    total = 0

    for features, arousal_labels, valence_labels in loader:
        features = features.to(device, non_blocking=True)
        arousal_targets = arousal_labels.to(device, non_blocking=True)
        valence_targets = valence_labels.to(device, non_blocking=True)

        arousal_logits, valence_logits = model(features)
        correct_arousal += (arousal_logits.argmax(1) == arousal_targets).sum().item()
        correct_valence += (valence_logits.argmax(1) == valence_targets).sum().item()
        total += features.size(0)

    arousal_acc = correct_arousal / total if total > 0 else 0.0
    valence_acc = correct_valence / total if total > 0 else 0.0
    return {
        "arousal_acc": arousal_acc,
        "valence_acc": valence_acc,
        "avg_acc": (arousal_acc + valence_acc) / 2,
    }


def train_fold_eegnet(
    train_loader: DataLoader[tuple[torch.Tensor, int, int]],
    val_loader: DataLoader[tuple[torch.Tensor, int, int]],
    *,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[EEGNetClassifier, dict[str, float]]:
    """Train EEGNet on DE features for one fold."""
    model = EEGNetClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=config.patience)
    use_amp = device.type == "cuda"

    best_val_acc = 0.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_total = 0

        for features, arousal_labels, valence_labels in train_loader:
            features = features.to(device, non_blocking=True)
            arousal_targets = arousal_labels.to(device, non_blocking=True)
            valence_targets = valence_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                arousal_logits, valence_logits = model(features)
                loss = criterion(arousal_logits, arousal_targets) + criterion(valence_logits, valence_targets)

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_total += features.size(0)

        scheduler.step()
        val_metrics = _evaluate(model, val_loader, device)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch + 1}/{config.epochs} | "
                f"Loss: {train_loss / train_total:.4f} | "
                f"Val A/V: {val_metrics['arousal_acc']:.3f}/{val_metrics['valence_acc']:.3f}"
            )

        if val_metrics["avg_acc"] > best_val_acc:
            best_val_acc = val_metrics["avg_acc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if not config.no_early_stop and stopper.step(val_metrics["avg_acc"]):
            logger.info(f"  Early stopping at epoch {epoch + 1} (patience={config.patience})")
            break

    model.load_state_dict(best_state)
    return model, _evaluate(model, val_loader, device)


def train_fold_pretrained(
    train_loader: DataLoader[tuple[torch.Tensor, int, int]],
    val_loader: DataLoader[tuple[torch.Tensor, int, int]],
    *,
    base_model: PretrainedDualHead,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[PretrainedDualHead, dict[str, float]]:
    """Train pretrained model for one fold with two-phase strategy.

    Phase 1 (first 1/3 epochs): frozen backbone, train heads only with warmup.
    Phase 2 (remaining epochs): unfreeze backbone, full fine-tuning with warmup + cosine decay.

    Reinitializes heads from scratch each fold but reuses the pretrained backbone
    weights (avoids re-downloading).
    """
    # Deep copy so each fold starts fresh, reusing cached backbone weights
    model = copy.deepcopy(base_model).to(device)
    model.freeze_backbone()
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"

    phase1_epochs = max(1, config.epochs // 3)
    phase2_epochs = config.epochs - phase1_epochs

    best_val_acc = 0.0
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

        for features, arousal_labels, valence_labels in train_loader:
            features = features.to(device, non_blocking=True)
            arousal_targets = arousal_labels.to(device, non_blocking=True)
            valence_targets = valence_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                arousal_logits, valence_logits = model(features)
                loss = criterion(arousal_logits, arousal_targets) + criterion(valence_logits, valence_targets)

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_total += features.size(0)

        scheduler.step()
        val_metrics = _evaluate(model, val_loader, device)
        elapsed = time.monotonic() - epoch_start
        logger.info(
            f"  [Phase 1] Epoch {epoch + 1}/{phase1_epochs} | "
            f"Loss: {train_loss / train_total:.4f} | "
            f"Val A/V: {val_metrics['arousal_acc']:.3f}/{val_metrics['valence_acc']:.3f} | "
            f"{elapsed:.1f}s"
        )

        if val_metrics["avg_acc"] > best_val_acc:
            best_val_acc = val_metrics["avg_acc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if not config.no_early_stop and stopper.step(val_metrics["avg_acc"]):
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

        for features, arousal_labels, valence_labels in train_loader:
            features = features.to(device, non_blocking=True)
            arousal_targets = arousal_labels.to(device, non_blocking=True)
            valence_targets = valence_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                arousal_logits, valence_logits = model(features)
                loss = criterion(arousal_logits, arousal_targets) + criterion(valence_logits, valence_targets)

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_total += features.size(0)

        scheduler.step()
        val_metrics = _evaluate(model, val_loader, device)
        elapsed = time.monotonic() - epoch_start

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  [Phase 2] Epoch {epoch + 1}/{phase2_epochs} | "
                f"Loss: {train_loss / train_total:.4f} | "
                f"Val A/V: {val_metrics['arousal_acc']:.3f}/{val_metrics['valence_acc']:.3f} | "
                f"{elapsed:.1f}s"
            )

        if val_metrics["avg_acc"] > best_val_acc:
            best_val_acc = val_metrics["avg_acc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if not config.no_early_stop and stopper.step(val_metrics["avg_acc"]):
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

    # Backend optimizations
    torch.set_float32_matmul_precision("high")
    # cudnn.benchmark is safe here: LOSO feeds fixed-shape batches
    # (batch × 32 × 800 for CBraMod, batch × 160 for EEGNet), so cudnn
    # picks one kernel on first iter and sticks with it — deterministic
    # for a given seed + shape, and ~10-15% faster on transformer-ish ops.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if config.batch_size is None:
        config.batch_size = _default_batch_size_for(device)
        logger.info(f"Auto-selected batch_size={config.batch_size} for device={device.type}")

    logger.info(f"Using device: {device}")
    logger.info(
        f"Config: model={config.model_type}, cv={config.cv_mode}, "
        f"epochs={config.epochs}, lr={config.lr}, batch_size={config.batch_size}"
    )

    mode: Literal["features", "raw"] = "raw" if config.model_type == "cbramod" else "features"

    try:
        dataset = load_dataset(mode=mode)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

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

    all_metrics: list[dict[str, float]] = []
    best_overall_acc = 0.0
    best_model_state: dict[str, torch.Tensor] | None = None
    dl_kwargs = _dataloader_kwargs(device)
    train_start = time.monotonic()

    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        logger.info(f"\n--- Fold {fold_idx + 1}/{len(splits)} (train={len(train_indices)}, val={len(val_indices)}) ---")

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        train_loader: DataLoader[tuple[torch.Tensor, int, int]] = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True,
            **dl_kwargs,  # type: ignore[arg-type]
        )
        val_loader: DataLoader[tuple[torch.Tensor, int, int]] = DataLoader(
            val_subset,
            batch_size=config.batch_size,
            shuffle=False,
            **dl_kwargs,  # type: ignore[arg-type]
        )

        fold_model: EEGModel
        if config.model_type == "cbramod":
            assert base_pretrained is not None
            fold_model, metrics = train_fold_pretrained(
                train_loader,
                val_loader,
                base_model=base_pretrained,
                config=config,
                device=device,
            )
        else:
            fold_model, metrics = train_fold_eegnet(
                train_loader,
                val_loader,
                config=config,
                device=device,
            )

        all_metrics.append(metrics)

        # Track best fold for checkpoint
        if metrics["avg_acc"] > best_overall_acc:
            best_overall_acc = metrics["avg_acc"]
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
        n = len(splits)
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_name = "cbramod_best.pt" if config.model_type == "cbramod" else "eegnet_best.pt"
        checkpoint_path = CHECKPOINTS_DIR / checkpoint_name
        torch.save(
            {
                "model_type": config.model_type,
                "model_state_dict": best_model_state,
                "metrics": {
                    "avg_arousal_acc": sum(m["arousal_acc"] for m in all_metrics) / n,
                    "avg_valence_acc": sum(m["valence_acc"] for m in all_metrics) / n,
                    "avg_overall_acc": sum(m["avg_acc"] for m in all_metrics) / n,
                    "best_fold_acc": best_overall_acc,
                    "n_folds": n,
                    "cv_mode": config.cv_mode,
                    "epochs": config.epochs,
                },
                "config": {
                    "model_type": config.model_type,
                    "cv_mode": config.cv_mode,
                    "epochs": config.epochs,
                    "lr": config.lr,
                    "finetune_lr": config.finetune_lr,
                    "batch_size": config.batch_size,
                    "weight_decay": config.weight_decay,
                    "max_grad_norm": config.max_grad_norm,
                    "patience": config.patience,
                    "seed": config.seed,
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


def _print_results_table(
    all_metrics: list[dict[str, float]],
    *,
    cv_mode: str,
    total_time: float,
) -> None:
    """Print per-fold results with mean ± std and wall-clock time."""
    n = len(all_metrics)
    logger.info(f"\n{'=' * 66}")
    logger.info(f"Cross-Validation Results ({n} folds, {cv_mode})")
    logger.info(f"{'=' * 66}")
    logger.info(f"{'Fold':<6} | {'Arousal Acc':>11} | {'Valence Acc':>11} | {'Avg Acc':>7}")
    logger.info(f"{'-' * 6}-+-{'-' * 11}-+-{'-' * 11}-+-{'-' * 7}")

    for i, m in enumerate(all_metrics):
        logger.info(f"{i + 1:<6} | {m['arousal_acc']:>11.4f} | {m['valence_acc']:>11.4f} | {m['avg_acc']:>7.4f}")

    avg_arousal = sum(m["arousal_acc"] for m in all_metrics) / n
    avg_valence = sum(m["valence_acc"] for m in all_metrics) / n
    avg_overall = sum(m["avg_acc"] for m in all_metrics) / n
    std_arousal = _std([m["arousal_acc"] for m in all_metrics])
    std_valence = _std([m["valence_acc"] for m in all_metrics])
    std_overall = _std([m["avg_acc"] for m in all_metrics])

    logger.info(f"{'-' * 6}-+-{'-' * 11}-+-{'-' * 11}-+-{'-' * 7}")
    logger.info(f"{'Mean':<6} | {avg_arousal:>11.4f} | {avg_valence:>11.4f} | {avg_overall:>7.4f}")
    logger.info(f"{'Std':<6} | {std_arousal:>11.4f} | {std_valence:>11.4f} | {std_overall:>7.4f}")

    minutes, seconds = divmod(total_time, 60)
    logger.info(f"\nTotal training time: {int(minutes)}m {seconds:.1f}s")


def _print_comparison_table(results: dict[str, dict[str, float]], source: str) -> None:
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Model Comparison on {source.upper()}")
    logger.info(f"{'=' * 60}")
    logger.info(f"{'Model':<16} | {'Arousal Acc':>11} | {'Valence Acc':>11} | {'Avg Acc':>7}")
    logger.info(f"{'-' * 16}-+-{'-' * 11}-+-{'-' * 11}-+-{'-' * 7}")
    for name, metrics in results.items():
        label = "EEGNet (DE)" if name == "eegnet" else "CBraMod (FT)"
        logger.info(
            f"{label:<16} | {metrics['arousal_acc']:>11.3f} | "
            f"{metrics['valence_acc']:>11.3f} | {metrics['avg_acc']:>7.3f}"
        )


def compare(config: TrainingConfig, *, retrain: bool = False) -> None:
    """Compare EEGNet and CBraMod metrics.

    By default, loads metrics from existing checkpoints. Use retrain=True
    to train both models from scratch.
    """
    if not retrain:
        # Try loading metrics from existing checkpoints
        results: dict[str, dict[str, float]] = {}
        checkpoint_map = {
            "eegnet": CHECKPOINTS_DIR / "eegnet_best.pt",
            "cbramod": CHECKPOINTS_DIR / "cbramod_best.pt",
        }

        for model_type, path in checkpoint_map.items():
            if path.exists():
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                m = checkpoint.get("metrics", {})
                results[model_type] = {
                    "arousal_acc": m.get("avg_arousal_acc", 0.0),
                    "valence_acc": m.get("avg_valence_acc", 0.0),
                    "avg_acc": m.get("avg_overall_acc", 0.0),
                }
                info = f"cv={m.get('cv_mode', '?')}, {m.get('n_folds', '?')} folds, {m.get('epochs', '?')} epochs"
                logger.info(f"Loaded {model_type} checkpoint ({info})")
            else:
                logger.warning(f"No checkpoint found for {model_type} at {path}")

        if len(results) >= 2:
            _print_comparison_table(results, "deap")
            return
        elif results:
            logger.info("Only one checkpoint found. Retraining missing model(s)...\n")
        else:
            logger.info("No checkpoints found. Training both models...\n")

    # Full retraining path
    if config.seed is not None:
        _set_seed(config.seed)

    device = _get_device()
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if config.batch_size is None:
        config.batch_size = _default_batch_size_for(device)
        logger.info(f"Auto-selected batch_size={config.batch_size} for device={device.type}")

    results = {}

    for model_type in ("eegnet", "cbramod"):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training {model_type}...")
        logger.info(f"{'=' * 60}\n")

        mode: Literal["features", "raw"] = "raw" if model_type == "cbramod" else "features"
        dataset = load_dataset(mode=mode)

        if config.cv_mode == "loso":
            splits = make_loso_splits(dataset, max_folds=config.max_folds)
        else:
            splits = make_grouped_splits(dataset, n_folds=config.n_folds)

        all_metrics: list[dict[str, float]] = []

        # Create a per-model config so fold functions see the right model_type
        model_config = TrainingConfig(
            model_type=model_type,  # type: ignore[arg-type]
            cv_mode=config.cv_mode,
            epochs=config.epochs,
            lr=config.lr,
            finetune_lr=config.finetune_lr,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,
            patience=config.patience,
            seed=config.seed,
            no_early_stop=config.no_early_stop,
        )

        base_pretrained: PretrainedDualHead | None = None
        if model_type == "cbramod":
            base_pretrained = load_pretrained_dual_head()

        dl_kwargs = _dataloader_kwargs(device)

        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            logger.info(f"--- Fold {fold_idx + 1}/{len(splits)} ---")
            train_loader: DataLoader[tuple[torch.Tensor, int, int]] = DataLoader(
                Subset(dataset, train_indices),
                batch_size=config.batch_size,
                shuffle=True,
                **dl_kwargs,  # type: ignore[arg-type]
            )
            val_loader: DataLoader[tuple[torch.Tensor, int, int]] = DataLoader(
                Subset(dataset, val_indices),
                batch_size=config.batch_size,
                shuffle=False,
                **dl_kwargs,  # type: ignore[arg-type]
            )

            fold_model: EEGModel
            if model_type == "cbramod":
                assert base_pretrained is not None
                fold_model, metrics = train_fold_pretrained(
                    train_loader,
                    val_loader,
                    base_model=base_pretrained,
                    config=model_config,
                    device=device,
                )
            else:
                fold_model, metrics = train_fold_eegnet(
                    train_loader,
                    val_loader,
                    config=model_config,
                    device=device,
                )
            all_metrics.append(metrics)
            del fold_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        results[model_type] = {
            "arousal_acc": sum(m["arousal_acc"] for m in all_metrics) / len(splits),
            "valence_acc": sum(m["valence_acc"] for m in all_metrics) / len(splits),
            "avg_acc": sum(m["avg_acc"] for m in all_metrics) / len(splits),
        }

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
