"""Training script for the dual-head EEGNet classifier.

Trains with binary cross-entropy on both arousal and valence heads,
using 5-fold cross-validation on synthetic or DEAP EEG data.

Usage:
    uv run train-model
    uv run train-model --epochs 50 --lr 0.001
"""

from __future__ import annotations

import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from cortexdj.core.paths import CHECKPOINTS_DIR, SYNTHETIC_DATA_DIR
from cortexdj.ml.dataset import EEGEmotionDataset
from cortexdj.ml.model import EEGNetClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def train_fold(
    train_loader: DataLoader[tuple[torch.Tensor, int, int]],
    val_loader: DataLoader[tuple[torch.Tensor, int, int]],
    *,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[EEGNetClassifier, dict[str, float]]:
    model = EEGNetClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = model.state_dict()

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct_arousal = 0
        train_correct_valence = 0
        train_total = 0

        for features, arousal_labels, valence_labels in train_loader:
            features = features.to(device)
            arousal_targets = torch.tensor(arousal_labels, dtype=torch.long, device=device)
            valence_targets = torch.tensor(valence_labels, dtype=torch.long, device=device)

            optimizer.zero_grad()
            arousal_logits, valence_logits = model(features)

            loss_arousal = criterion(arousal_logits, arousal_targets)
            loss_valence = criterion(valence_logits, valence_targets)
            loss = loss_arousal + loss_valence

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_correct_arousal += (arousal_logits.argmax(1) == arousal_targets).sum().item()
            train_correct_valence += (valence_logits.argmax(1) == valence_targets).sum().item()
            train_total += features.size(0)

        # Validation
        model.eval()
        val_correct_arousal = 0
        val_correct_valence = 0
        val_total = 0

        with torch.no_grad():
            for features, arousal_labels, valence_labels in val_loader:
                features = features.to(device)
                arousal_targets = torch.tensor(arousal_labels, dtype=torch.long, device=device)
                valence_targets = torch.tensor(valence_labels, dtype=torch.long, device=device)

                arousal_logits, valence_logits = model(features)
                val_correct_arousal += (arousal_logits.argmax(1) == arousal_targets).sum().item()
                val_correct_valence += (valence_logits.argmax(1) == valence_targets).sum().item()
                val_total += features.size(0)

        val_arousal_acc = val_correct_arousal / val_total if val_total > 0 else 0
        val_valence_acc = val_correct_valence / val_total if val_total > 0 else 0
        val_avg_acc = (val_arousal_acc + val_valence_acc) / 2

        if (epoch + 1) % 5 == 0 or epoch == 0:
            train_arousal_acc = train_correct_arousal / train_total if train_total > 0 else 0
            train_valence_acc = train_correct_valence / train_total if train_total > 0 else 0
            logger.info(
                f"  Epoch {epoch + 1}/{epochs} | "
                f"Loss: {train_loss / train_total:.4f} | "
                f"Train A/V: {train_arousal_acc:.3f}/{train_valence_acc:.3f} | "
                f"Val A/V: {val_arousal_acc:.3f}/{val_valence_acc:.3f}"
            )

        if val_avg_acc > best_val_acc:
            best_val_acc = val_avg_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    metrics = {
        "best_val_arousal_acc": val_arousal_acc,
        "best_val_valence_acc": val_valence_acc,
        "best_val_avg_acc": best_val_acc,
    }
    return model, metrics


def train(
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
    n_folds: int = 5,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if not SYNTHETIC_DATA_DIR.exists():
        logger.error(f"Data directory not found: {SYNTHETIC_DATA_DIR}")
        logger.error("Run `uv run generate-synthetic` first to create training data.")
        return

    logger.info(f"Loading dataset from {SYNTHETIC_DATA_DIR}...")
    dataset = EEGEmotionDataset(SYNTHETIC_DATA_DIR)
    logger.info(f"Loaded {len(dataset)} segments")

    if len(dataset) == 0:
        logger.error("No data found. Run `uv run generate-synthetic` first.")
        return

    n_samples = len(dataset)
    indices = list(range(n_samples))
    fold_size = n_samples // n_folds
    all_metrics: list[dict[str, float]] = []

    for fold in range(n_folds):
        logger.info(f"\n--- Fold {fold + 1}/{n_folds} ---")

        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
        val_indices = indices[val_start:val_end]
        train_indices = indices[:val_start] + indices[val_end:]

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model, metrics = train_fold(
            train_loader,
            val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
        )
        all_metrics.append(metrics)

    avg_arousal = sum(m["best_val_arousal_acc"] for m in all_metrics) / n_folds
    avg_valence = sum(m["best_val_valence_acc"] for m in all_metrics) / n_folds
    avg_overall = sum(m["best_val_avg_acc"] for m in all_metrics) / n_folds
    logger.info(f"\n=== Cross-Validation Results ({n_folds} folds) ===")
    logger.info(f"Avg Arousal Accuracy: {avg_arousal:.4f}")
    logger.info(f"Avg Valence Accuracy: {avg_valence:.4f}")
    logger.info(f"Avg Overall Accuracy: {avg_overall:.4f}")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINTS_DIR / "eegnet_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": {
                "avg_arousal_acc": avg_arousal,
                "avg_valence_acc": avg_valence,
                "avg_overall_acc": avg_overall,
                "n_folds": n_folds,
                "epochs": epochs,
            },
        },
        checkpoint_path,
    )
    logger.info(f"Saved best model to {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EEGNet on EEG emotion data")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per fold")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, n_folds=args.folds)
