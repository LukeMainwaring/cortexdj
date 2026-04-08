"""Training script for EEG emotion classifiers.

Supports training the custom EEGNet (on DE features) or fine-tuning
CBraMod pretrained model (on raw EEG), with LOSO or grouped cross-validation.

Usage:
    uv run train-model
    uv run train-model --source deap --model cbramod --cv loso
    uv run compare-models
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from cortexdj.core.paths import CHECKPOINTS_DIR
from cortexdj.ml.dataset import (
    DEAPFeatureDataset,
    DEAPRawDataset,
    EEGEmotionDataset,
    load_dataset,
)
from cortexdj.ml.model import EEGNetClassifier
from cortexdj.ml.pretrained import PretrainedDualHead, load_pretrained_dual_head

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Type for any dataset that has participant_ids
type ParticipantDataset = EEGEmotionDataset | DEAPFeatureDataset | DEAPRawDataset
type EEGModel = EEGNetClassifier | PretrainedDualHead


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

    with torch.no_grad():
        for features, arousal_labels, valence_labels in loader:
            features = features.to(device)
            arousal_targets = arousal_labels.clone().detach().to(dtype=torch.long, device=device)
            valence_targets = valence_labels.clone().detach().to(dtype=torch.long, device=device)

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
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[EEGNetClassifier, dict[str, float]]:
    """Train EEGNet on DE features for one fold."""
    model = EEGNetClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state: dict[str, torch.Tensor] = {}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_total = 0

        for features, arousal_labels, valence_labels in train_loader:
            features = features.to(device)
            arousal_targets = arousal_labels.clone().detach().to(dtype=torch.long, device=device)
            valence_targets = valence_labels.clone().detach().to(dtype=torch.long, device=device)

            optimizer.zero_grad()
            arousal_logits, valence_logits = model(features)

            loss = criterion(arousal_logits, arousal_targets) + criterion(valence_logits, valence_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_total += features.size(0)

        val_metrics = _evaluate(model, val_loader, device)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch + 1}/{epochs} | "
                f"Loss: {train_loss / train_total:.4f} | "
                f"Val A/V: {val_metrics['arousal_acc']:.3f}/{val_metrics['valence_acc']:.3f}"
            )

        if val_metrics["avg_acc"] > best_val_acc:
            best_val_acc = val_metrics["avg_acc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, _evaluate(model, val_loader, device)


def train_fold_pretrained(
    train_loader: DataLoader[tuple[torch.Tensor, int, int]],
    val_loader: DataLoader[tuple[torch.Tensor, int, int]],
    *,
    base_model: PretrainedDualHead,
    epochs: int,
    head_lr: float,
    finetune_lr: float,
    device: torch.device,
) -> tuple[PretrainedDualHead, dict[str, float]]:
    """Train pretrained model for one fold with two-phase strategy.

    Phase 1 (first 1/3 epochs): frozen backbone, train heads only.
    Phase 2 (remaining epochs): unfreeze backbone, full fine-tuning with lower LR.

    Reinitializes heads from scratch each fold but reuses the pretrained backbone
    weights (avoids re-downloading).
    """
    # Deep copy so each fold starts fresh, reusing cached backbone weights
    import copy

    model = copy.deepcopy(base_model).to(device)
    model.freeze_backbone()
    criterion = nn.CrossEntropyLoss()

    phase1_epochs = max(1, epochs // 3)
    phase2_epochs = epochs - phase1_epochs

    best_val_acc = 0.0
    best_state: dict[str, torch.Tensor] = {}

    # Phase 1: frozen backbone
    head_params = list(model.arousal_head.parameters()) + list(model.valence_head.parameters())
    optimizer = torch.optim.Adam(head_params, lr=head_lr)

    for epoch in range(phase1_epochs):
        epoch_start = time.monotonic()
        model.train()
        train_loss = 0.0
        train_total = 0

        for features, arousal_labels, valence_labels in train_loader:
            features = features.to(device)
            arousal_targets = arousal_labels.clone().detach().to(dtype=torch.long, device=device)
            valence_targets = valence_labels.clone().detach().to(dtype=torch.long, device=device)

            optimizer.zero_grad()
            arousal_logits, valence_logits = model(features)

            loss = criterion(arousal_logits, arousal_targets) + criterion(valence_logits, valence_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_total += features.size(0)

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

    # Phase 2: full fine-tuning
    model.unfreeze_backbone()
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)

    for epoch in range(phase2_epochs):
        epoch_start = time.monotonic()
        model.train()
        train_loss = 0.0
        train_total = 0

        for features, arousal_labels, valence_labels in train_loader:
            features = features.to(device)
            arousal_targets = arousal_labels.clone().detach().to(dtype=torch.long, device=device)
            valence_targets = valence_labels.clone().detach().to(dtype=torch.long, device=device)

            optimizer.zero_grad()
            arousal_logits, valence_logits = model(features)

            loss = criterion(arousal_logits, arousal_targets) + criterion(valence_logits, valence_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_total += features.size(0)

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

    model.load_state_dict(best_state)
    return model, _evaluate(model, val_loader, device)


def train(
    *,
    source: Literal["synthetic", "deap"] = "synthetic",
    model_type: Literal["eegnet", "cbramod"] = "eegnet",
    cv_mode: Literal["loso", "grouped"] = "grouped",
    epochs: int = 30,
    lr: float = 1e-3,
    finetune_lr: float = 1e-5,
    batch_size: int = 32,
    n_folds: int = 5,
    max_folds: int | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Config: source={source}, model={model_type}, cv={cv_mode}")

    mode: Literal["features", "raw"] = "raw" if model_type == "cbramod" else "features"

    try:
        dataset = load_dataset(source=source, mode=mode)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    logger.info(f"Loaded {len(dataset)} segments from {source} ({mode} mode)")

    if len(dataset) == 0:
        logger.error("No data found.")
        return

    # Build CV splits
    if cv_mode == "loso":
        splits = make_loso_splits(dataset, max_folds=max_folds)
    else:
        splits = make_grouped_splits(dataset, n_folds=n_folds)

    logger.info(f"Cross-validation: {len(splits)} folds ({cv_mode})")

    # Pre-load pretrained model once (avoids re-downloading per fold)
    base_pretrained: PretrainedDualHead | None = None
    if model_type == "cbramod":
        logger.info("Loading pretrained CBraMod backbone (one-time)...")
        load_start = time.monotonic()
        base_pretrained = load_pretrained_dual_head()
        logger.info(f"Pretrained model loaded in {time.monotonic() - load_start:.1f}s")

    all_metrics: list[dict[str, float]] = []
    best_overall_acc = 0.0
    best_model_state: dict[str, torch.Tensor] | None = None

    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        logger.info(f"\n--- Fold {fold_idx + 1}/{len(splits)} (train={len(train_indices)}, val={len(val_indices)}) ---")

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        train_loader: DataLoader[tuple[torch.Tensor, int, int]] = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True
        )
        val_loader: DataLoader[tuple[torch.Tensor, int, int]] = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False
        )

        fold_model: EEGModel
        if model_type == "cbramod":
            assert base_pretrained is not None
            fold_model, metrics = train_fold_pretrained(
                train_loader,
                val_loader,
                base_model=base_pretrained,
                epochs=epochs,
                head_lr=lr,
                finetune_lr=finetune_lr,
                device=device,
            )
        else:
            fold_model, metrics = train_fold_eegnet(
                train_loader,
                val_loader,
                epochs=epochs,
                lr=lr,
                device=device,
            )

        all_metrics.append(metrics)

        # Track best fold for checkpoint
        if metrics["avg_acc"] > best_overall_acc:
            best_overall_acc = metrics["avg_acc"]
            best_model_state = {k: v.cpu().clone() for k, v in fold_model.state_dict().items()}

    # Report cross-validation results
    avg_arousal = sum(m["arousal_acc"] for m in all_metrics) / len(splits)
    avg_valence = sum(m["valence_acc"] for m in all_metrics) / len(splits)
    avg_overall = sum(m["avg_acc"] for m in all_metrics) / len(splits)
    logger.info(f"\n=== Cross-Validation Results ({len(splits)} folds, {cv_mode}) ===")
    logger.info(f"Avg Arousal Accuracy: {avg_arousal:.4f}")
    logger.info(f"Avg Valence Accuracy: {avg_valence:.4f}")
    logger.info(f"Avg Overall Accuracy: {avg_overall:.4f}")

    # Save best fold's model
    if best_model_state is not None:
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_name = "cbramod_best.pt" if model_type == "cbramod" else "eegnet_best.pt"
        checkpoint_path = CHECKPOINTS_DIR / checkpoint_name
        torch.save(
            {
                "model_type": model_type,
                "model_state_dict": best_model_state,
                "metrics": {
                    "avg_arousal_acc": avg_arousal,
                    "avg_valence_acc": avg_valence,
                    "avg_overall_acc": avg_overall,
                    "best_fold_acc": best_overall_acc,
                    "n_folds": len(splits),
                    "cv_mode": cv_mode,
                    "source": source,
                    "epochs": epochs,
                },
            },
            checkpoint_path,
        )
        logger.info(f"Saved best model to {checkpoint_path}")


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


def compare(
    *,
    source: Literal["deap"] = "deap",
    retrain: bool = False,
    epochs: int = 30,
    cv_mode: Literal["loso", "grouped"] = "loso",
    max_folds: int | None = None,
) -> None:
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
                info = (
                    f"source={m.get('source', '?')}, "
                    f"cv={m.get('cv_mode', '?')}, "
                    f"{m.get('n_folds', '?')} folds, "
                    f"{m.get('epochs', '?')} epochs"
                )
                logger.info(f"Loaded {model_type} checkpoint ({info})")
            else:
                logger.warning(f"No checkpoint found for {model_type} at {path}")

        if len(results) >= 2:
            _print_comparison_table(results, source)
            return
        elif results:
            logger.info("Only one checkpoint found. Retraining missing model(s)...\n")
        else:
            logger.info("No checkpoints found. Training both models...\n")

    # Full retraining path
    results = {}

    for model_type in ("eegnet", "cbramod"):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training {model_type}...")
        logger.info(f"{'=' * 60}\n")

        mode: Literal["features", "raw"] = "raw" if model_type == "cbramod" else "features"
        dataset = load_dataset(source=source, mode=mode)

        if cv_mode == "loso":
            splits = make_loso_splits(dataset, max_folds=max_folds)
        else:
            splits = make_grouped_splits(dataset)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_metrics: list[dict[str, float]] = []

        base_pretrained: PretrainedDualHead | None = None
        if model_type == "cbramod":
            base_pretrained = load_pretrained_dual_head()

        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            logger.info(f"--- Fold {fold_idx + 1}/{len(splits)} ---")
            train_loader: DataLoader[tuple[torch.Tensor, int, int]] = DataLoader(
                Subset(dataset, train_indices), batch_size=32, shuffle=True
            )
            val_loader: DataLoader[tuple[torch.Tensor, int, int]] = DataLoader(
                Subset(dataset, val_indices), batch_size=32, shuffle=False
            )

            if model_type == "cbramod":
                assert base_pretrained is not None
                _, metrics = train_fold_pretrained(
                    train_loader,
                    val_loader,
                    base_model=base_pretrained,
                    epochs=epochs,
                    head_lr=1e-3,
                    finetune_lr=1e-5,
                    device=device,
                )
            else:
                _, metrics = train_fold_eegnet(
                    train_loader,
                    val_loader,
                    epochs=epochs,
                    lr=1e-3,
                    device=device,
                )
            all_metrics.append(metrics)

        results[model_type] = {
            "arousal_acc": sum(m["arousal_acc"] for m in all_metrics) / len(splits),
            "valence_acc": sum(m["valence_acc"] for m in all_metrics) / len(splits),
            "avg_acc": sum(m["avg_acc"] for m in all_metrics) / len(splits),
        }

    _print_comparison_table(results, source)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EEG emotion classifier")
    parser.add_argument(
        "--source",
        choices=["synthetic", "deap"],
        default="synthetic",
        help="Dataset source (default: synthetic)",
    )
    parser.add_argument(
        "--model",
        choices=["eegnet", "cbramod"],
        default="eegnet",
        help="Model type (default: eegnet)",
    )
    parser.add_argument(
        "--cv",
        choices=["loso", "grouped"],
        default="grouped",
        help="Cross-validation strategy (default: grouped)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per fold")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=1e-5,
        help="Fine-tuning LR for pretrained backbone (default: 1e-5)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (grouped mode)")
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Limit LOSO folds for faster development",
    )
    args = parser.parse_args()

    train(
        source=args.source,
        model_type=args.model,
        cv_mode=args.cv,
        epochs=args.epochs,
        lr=args.lr,
        finetune_lr=args.finetune_lr,
        batch_size=args.batch_size,
        n_folds=args.folds,
        max_folds=args.max_folds,
    )


def compare_main() -> None:
    parser = argparse.ArgumentParser(description="Compare EEGNet vs CBraMod on DEAP")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining instead of loading existing checkpoints",
    )
    parser.add_argument(
        "--cv",
        choices=["loso", "grouped"],
        default="loso",
        help="Cross-validation strategy for retraining (default: loso)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per fold (retraining only)")
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Limit LOSO folds for faster development (retraining only)",
    )
    args = parser.parse_args()

    compare(retrain=args.retrain, cv_mode=args.cv, epochs=args.epochs, max_folds=args.max_folds)
