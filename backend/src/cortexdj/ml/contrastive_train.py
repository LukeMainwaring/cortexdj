"""Training entrypoint for the EEG↔CLAP contrastive encoder.

Deterministic 24/4/4 subject split (or 3/1/1 in --quick mode). Full
fine-tuning of the CBraMod backbone alongside the projection MLP, with
differential learning rates. EMA-smoothed early stopping on validation
top-5 retrieval accuracy.

Run (local MPS / CPU):
  uv run --directory backend python -m cortexdj.ml.contrastive_train --quick

Run (Modal):
  modal run backend/scripts/modal_train.py --command train-contrastive
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PairBatch = tuple[torch.Tensor, torch.Tensor, int, int]

from cortexdj.core.paths import CHECKPOINTS_DIR
from cortexdj.ml.contrastive import (
    CLAP_MODEL_ID,
    EMBEDDING_DIM,
    EegCLAPEncoder,
    retrieval_metrics,
    symmetric_info_nce,
)
from cortexdj.ml.contrastive_dataset import DeapClapPairDataset
from cortexdj.ml.train import EarlyStopping, _get_device, _set_seed

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_FILENAME = "contrastive_best.pt"

DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_BACKBONE_LR = 1e-4
DEFAULT_PROJECTION_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_WARMUP_EPOCHS = 3
DEFAULT_PATIENCE = 8
DEFAULT_SEED = 42

QUICK_EPOCHS = 5
QUICK_TRAIN_SUBJECTS = 3
QUICK_VAL_SUBJECTS = 1
QUICK_TEST_SUBJECTS = 1

FULL_TRAIN_SUBJECTS = 24
FULL_VAL_SUBJECTS = 4
FULL_TEST_SUBJECTS = 4


@dataclasses.dataclass
class ContrastiveConfig:
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    backbone_lr: float = DEFAULT_BACKBONE_LR
    projection_lr: float = DEFAULT_PROJECTION_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS
    patience: int = DEFAULT_PATIENCE
    seed: int = DEFAULT_SEED
    quick: bool = False


def _split_subjects(
    all_subjects: list[int],
    *,
    quick: bool,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    shuffled = list(all_subjects)
    rng.shuffle(shuffled)

    n_train, n_val, n_test = (
        (QUICK_TRAIN_SUBJECTS, QUICK_VAL_SUBJECTS, QUICK_TEST_SUBJECTS)
        if quick
        else (FULL_TRAIN_SUBJECTS, FULL_VAL_SUBJECTS, FULL_TEST_SUBJECTS)
    )
    train = sorted(shuffled[:n_train])
    val = sorted(shuffled[n_train : n_train + n_val])
    test = sorted(shuffled[n_train + n_val : n_train + n_val + n_test])
    return train, val, test


def _chunked_forward(
    model: EegCLAPEncoder,
    loader: DataLoader[PairBatch],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward the full loader once and return concatenated embeddings."""
    eegs: list[torch.Tensor] = []
    audios: list[torch.Tensor] = []
    trial_ids: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for eeg, audio, tid, _sid in loader:
            eeg = eeg.to(device)
            eegs.append(model(eeg).cpu())
            audios.append(audio)
            trial_ids.append(tid)
    return torch.cat(eegs), torch.cat(audios), torch.cat(trial_ids)


def _evaluate(
    model: EegCLAPEncoder,
    loader: DataLoader[PairBatch],
    *,
    temperature: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    eeg_emb, audio_emb, trial_ids = _chunked_forward(model, loader, device)
    temperature_cpu = temperature.detach().cpu()
    loss = symmetric_info_nce(eeg_emb, audio_emb, trial_ids, temperature_cpu)
    metrics = retrieval_metrics(eeg_emb, audio_emb, trial_ids)
    metrics["loss"] = float(loss.item())
    return metrics


def _cosine_lr(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _format_metrics(m: dict[str, float]) -> str:
    return (
        f"loss={m['loss']:.4f}  top1={m['top1']:.3f}  top5={m['top5']:.3f}  top10={m['top10']:.3f}  mrr={m['mrr']:.3f}"
    )


def _train_one_epoch(
    *,
    model: EegCLAPEncoder,
    temperature: torch.nn.Parameter,
    loader: DataLoader[PairBatch],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lr_scale: float,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * lr_scale
    for eeg, audio, tid, _sid in loader:
        eeg = eeg.to(device)
        audio = audio.to(device)
        tid = tid.to(device)
        eeg_emb = model(eeg)
        audio_emb = F.normalize(audio, dim=-1)
        loss = symmetric_info_nce(eeg_emb, audio_emb, tid, temperature)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(1, n_batches)


def train(config: ContrastiveConfig) -> Path:
    _set_seed(config.seed)
    device = _get_device()
    logger.info(f"Device: {device}, config: {config}")

    dataset = DeapClapPairDataset()
    all_subjects = dataset.subject_ids()
    logger.info(f"{len(dataset)} samples across {len(all_subjects)} subjects")

    train_subj, val_subj, test_subj = _split_subjects(all_subjects, quick=config.quick, seed=config.seed)
    logger.info(f"Split: train={train_subj} val={val_subj} test={test_subj}")

    train_idx = dataset.indices_for_subjects(train_subj)
    val_idx = dataset.indices_for_subjects(val_subj)
    test_idx = dataset.indices_for_subjects(test_subj)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=config.batch_size, num_workers=0)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=config.batch_size, num_workers=0)

    model = EegCLAPEncoder().to(device)
    temperature = torch.nn.Parameter(torch.tensor(math.log(1.0 / 0.07), device=device))

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "base_lr": config.backbone_lr, "lr": config.backbone_lr},
            {"params": model.projection_parameters(), "base_lr": config.projection_lr, "lr": config.projection_lr},
            {"params": [temperature], "base_lr": config.projection_lr, "lr": config.projection_lr},
        ],
        weight_decay=config.weight_decay,
    )

    early = EarlyStopping(patience=config.patience, min_epochs=max(5, config.warmup_epochs + 2))
    best_state: dict[str, torch.Tensor] | None = None
    best_temperature: float = 0.0
    best_val: dict[str, float] = {"loss": float("inf"), "top1": 0.0, "top5": 0.0, "top10": 0.0, "mrr": 0.0}

    for epoch in range(1, config.epochs + 1):
        lr_scale = _cosine_lr(epoch - 1, config.epochs, config.warmup_epochs)
        t0 = time.monotonic()
        train_loss = _train_one_epoch(
            model=model,
            temperature=temperature,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            lr_scale=lr_scale,
        )
        val_metrics = _evaluate(model, val_loader, temperature=temperature, device=device)
        epoch_s = time.monotonic() - t0

        logger.info(
            f"Epoch {epoch:3d}/{config.epochs}  lr_scale={lr_scale:.3f}  "
            f"train_loss={train_loss:.4f}  val {_format_metrics(val_metrics)}  ({epoch_s:.1f}s)"
        )

        should_stop, is_new_best = early.step(val_metrics["top5"])
        if is_new_best:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_temperature = float(temperature.detach().cpu().item())
            best_val = val_metrics
        if should_stop:
            logger.info(f"Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _evaluate(model, test_loader, temperature=temperature, device=device)
    logger.info(f"Test {_format_metrics(test_metrics)}")

    checkpoint_path = CHECKPOINTS_DIR / CHECKPOINT_FILENAME
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "state_dict": best_state if best_state is not None else model.state_dict(),
            "temperature": best_temperature,
            "clap_model_id": CLAP_MODEL_ID,
            "embedding_dim": EMBEDDING_DIM,
            "config": dataclasses.asdict(config),
            "val_metrics": best_val,
            "test_metrics": test_metrics,
            "train_subjects": train_subj,
            "val_subjects": val_subj,
            "test_subjects": test_subj,
        },
        checkpoint_path,
    )
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="5 epochs × 3 train / 1 val / 1 test subjects")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--projection-lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = ContrastiveConfig(seed=args.seed, quick=args.quick)
    if args.quick:
        config.epochs = QUICK_EPOCHS
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.backbone_lr is not None:
        config.backbone_lr = args.backbone_lr
    if args.projection_lr is not None:
        config.projection_lr = args.projection_lr

    try:
        train(config)
    except Exception:
        logger.exception("Contrastive training failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
