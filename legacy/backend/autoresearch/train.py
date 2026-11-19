"""Autoresearch baseline: EEGNet on DEAP with a 15-min wall-clock budget.

THIS FILE IS AGENT-EDITABLE. Rewrite it however you like — change the
model, optimizer, loss, augmentation, training loop — as long as you
respect these four contracts:

    1. Load the fixed split via ``prepare.load_splits()``.
    2. Call ``prepare.evaluate(model, val_ds, device)`` for metrics.
    3. Stop training when ``prepare.WALL_CLOCK_BUDGET_SECONDS`` is hit.
    4. Print one final line ``FINAL_METRIC=<float>`` AND write
       ``metrics.json`` to ``$AUTORESEARCH_RUN_DIR``.

See program.md for the full set of rules and the idea bank.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from autoresearch.prepare import (
    NUM_BANDS,
    NUM_CHANNELS,
    WALL_CLOCK_BUDGET_SECONDS,
    evaluate,
    load_splits,
    pick_device,
    set_seeds,
)


class EEGNetClassifier(nn.Module):
    """Dual-head EEGNet-inspired classifier on DE features.

    Mirrors the production model at src/cortexdj/ml/model.py. Inlined here
    so the agent can rewrite architecture freely.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        spatial_filters: int = 32,
        temporal_filters: int = 64,
    ) -> None:
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, spatial_filters, kernel_size=(NUM_CHANNELS, 1)),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(spatial_filters, temporal_filters, kernel_size=(1, NUM_BANDS)),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.backbone = nn.Sequential(
            nn.Linear(temporal_filters, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.arousal_head = nn.Linear(hidden_dim, 2)
        self.valence_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.view(-1, 1, NUM_CHANNELS, NUM_BANDS)
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        x = x.view(x.size(0), -1)
        x = self.backbone(x)
        return self.arousal_head(x), self.valence_head(x)


def _class_weights(labels: npt.NDArray[np.int64], num_classes: int = 2) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes)[:num_classes]
    n_total = int(counts.sum())
    if n_total == 0:
        return torch.ones(num_classes, dtype=torch.float32)
    weights = n_total / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def _subset_labels(subset: torch.utils.data.Subset[object]) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    arousal = np.fromiter((subset[i][1] for i in range(len(subset))), dtype=np.int64, count=len(subset))
    valence = np.fromiter((subset[i][2] for i in range(len(subset))), dtype=np.int64, count=len(subset))
    return arousal, valence


def main() -> None:
    set_seeds()
    device = pick_device()
    budget = WALL_CLOCK_BUDGET_SECONDS
    print(f"[autoresearch] device={device} budget={budget}s", flush=True)

    train_ds, val_ds = load_splits()
    print(f"[autoresearch] train_n={len(train_ds)} val_n={len(val_ds)}", flush=True)

    a_labels, v_labels = _subset_labels(train_ds)
    a_w = _class_weights(a_labels).to(device)
    v_w = _class_weights(v_labels).to(device)

    model = EEGNetClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    arousal_loss = nn.CrossEntropyLoss(weight=a_w, label_smoothing=0.05)
    valence_loss = nn.CrossEntropyLoss(weight=v_w, label_smoothing=0.05)

    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    start = time.monotonic()
    best_f1 = 0.0
    step = 0
    epoch = 0
    eval_every = 100

    while time.monotonic() - start < budget:
        epoch += 1
        model.train()
        for features, arousal, valence in train_loader:
            if time.monotonic() - start >= budget:
                break
            features = features.to(device)
            arousal = arousal.to(device)
            valence = valence.to(device)

            a_logits, v_logits = model(features)
            loss = arousal_loss(a_logits, arousal) + valence_loss(v_logits, valence)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % eval_every == 0:
                eval_metrics = evaluate(model, val_ds, device)
                best_f1 = max(best_f1, eval_metrics["avg_macro_f1"])
                elapsed = time.monotonic() - start
                print(
                    f"[autoresearch] step={step} ep={epoch} "
                    f"elapsed={elapsed:.1f}s "
                    f"avg_f1={eval_metrics['avg_macro_f1']:.4f} "
                    f"best={best_f1:.4f} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )
                model.train()

    final = evaluate(model, val_ds, device)
    best_f1 = max(best_f1, final["avg_macro_f1"])

    out: dict[str, float | int | str] = {
        "avg_macro_f1": final["avg_macro_f1"],
        "arousal_f1": final["arousal_f1"],
        "valence_f1": final["valence_f1"],
        "best_avg_macro_f1": best_f1,
        "duration_s": round(time.monotonic() - start, 2),
        "steps": step,
        "epochs": epoch,
        "device": str(device),
    }

    run_dir = Path(os.environ.get("AUTORESEARCH_RUN_DIR", "/tmp/autoresearch_run"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2))

    # FINAL_METRIC is the number the agent optimizes. We use best-seen across
    # eval checkpoints (robust to end-of-run noise), matching how production
    # EEGNet picks its checkpoint.
    print(f"FINAL_METRIC={best_f1:.6f}", flush=True)


if __name__ == "__main__":
    main()
