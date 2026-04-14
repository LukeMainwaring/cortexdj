"""Training entrypoint for the EEG↔CLAP contrastive encoder.

Deterministic 24/4/4 subject split (or 3/1/1 in --quick mode). Full
fine-tuning of the CBraMod backbone alongside the projection MLP, with
differential learning rates. EMA-smoothed early stopping on validation
top-5 retrieval accuracy.

Observability: per-epoch TensorBoard scalars + a val-embedding projector
snapshot at the end of training (see backend/data/tensorboard_runs/).
Each epoch logs a timing breakdown (data / forward / backward / val) so
slowdowns are attributable to the right stage.

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
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cortexdj.core.paths import CHECKPOINTS_DIR, TENSORBOARD_RUNS_DIR
from cortexdj.ml.contrastive import (
    CLAP_MODEL_ID,
    EMBEDDING_DIM,
    EegCLAPEncoder,
    retrieval_metrics,
    symmetric_info_nce,
)
from cortexdj.ml.contrastive_dataset import (
    DeapClapPairDataset,
    build_audio_embedding_cache,
    load_resolved_stimuli,
)
from cortexdj.ml.train import EarlyStopping, _get_device, _set_seed

logger = logging.getLogger(__name__)

PairBatch = tuple[torch.Tensor, torch.Tensor, int, int]

CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_FILENAME = "contrastive_best.pt"

DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64
DEFAULT_BACKBONE_LR = 1e-4
DEFAULT_PROJECTION_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_WARMUP_EPOCHS = 3
DEFAULT_PATIENCE = 8
DEFAULT_GRAD_ACCUM = 1
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
    grad_accum: int = DEFAULT_GRAD_ACCUM
    seed: int = DEFAULT_SEED
    quick: bool = False
    use_tensorboard: bool = True


@dataclasses.dataclass
class EpochTiming:
    data_s: float = 0.0
    forward_s: float = 0.0
    backward_s: float = 0.0
    val_s: float = 0.0
    train_samples: int = 0

    def total_train_s(self) -> float:
        return self.data_s + self.forward_s + self.backward_s

    def samples_per_second(self) -> float:
        elapsed = self.total_train_s()
        return self.train_samples / elapsed if elapsed > 0 else 0.0

    def data_pct(self, epoch_s: float) -> float:
        return 100.0 * self.data_s / epoch_s if epoch_s > 0 else 0.0


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


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _chunked_forward(
    model: EegCLAPEncoder,
    loader: DataLoader[PairBatch],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward the loader once. Returns (eeg_emb, audio_emb, trial_ids, subject_ids)."""
    eegs: list[torch.Tensor] = []
    audios: list[torch.Tensor] = []
    trial_ids: list[torch.Tensor] = []
    subject_ids: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for eeg, audio, tid, sid in loader:
            eeg = eeg.to(device)
            eegs.append(model(eeg).cpu())
            audios.append(audio)
            trial_ids.append(tid)
            subject_ids.append(sid)
    return torch.cat(eegs), torch.cat(audios), torch.cat(trial_ids), torch.cat(subject_ids)


def _evaluate(
    model: EegCLAPEncoder,
    loader: DataLoader[PairBatch],
    *,
    temperature: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    eeg_emb, audio_emb, trial_ids, _subject_ids = _chunked_forward(model, loader, device)
    temperature_cpu = temperature.detach().cpu()
    loss = symmetric_info_nce(eeg_emb, audio_emb, trial_ids, temperature_cpu)
    metrics = retrieval_metrics(eeg_emb, audio_emb, trial_ids)
    metrics["loss"] = float(loss.item())
    return metrics


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
    grad_accum: int,
) -> tuple[float, EpochTiming]:
    """Run one training epoch. Returns (mean un-normalized loss, timing).

    `grad_accum` divides the per-batch loss so that accumulating gradients
    over N batches matches a single `optimizer.step()` on the mean loss.
    Note: this does NOT increase the in-batch contrastive negative pool —
    InfoNCE negatives are still batch_size per micro-batch. The benefit is
    smoother gradients, not more negatives.
    """
    model.train()
    timing = EpochTiming()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    batch_start = time.monotonic()
    for step_idx, (eeg, audio, tid, _sid) in enumerate(loader):
        timing.data_s += time.monotonic() - batch_start

        fwd_start = time.monotonic()
        eeg = eeg.to(device)
        audio = audio.to(device)
        tid = tid.to(device)
        eeg_emb = model(eeg)
        audio_emb = F.normalize(audio, dim=-1)
        loss = symmetric_info_nce(eeg_emb, audio_emb, tid, temperature)
        scaled_loss = loss / grad_accum
        timing.forward_s += time.monotonic() - fwd_start

        bwd_start = time.monotonic()
        scaled_loss.backward()  # type: ignore[no-untyped-call]
        is_accum_boundary = (step_idx + 1) % grad_accum == 0 or (step_idx + 1) == len(loader)
        if is_accum_boundary:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        timing.backward_s += time.monotonic() - bwd_start

        total_loss += float(loss.item())  # track un-normalized for display
        n_batches += 1
        timing.train_samples += eeg.shape[0]
        batch_start = time.monotonic()

    return total_loss / max(1, n_batches), timing


def _make_writer(config: ContrastiveConfig) -> Any | None:
    if not config.use_tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logger.warning("tensorboard not installed, skipping logging")
        return None

    run_name = f"contrastive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = TENSORBOARD_RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"TensorBoard logging to {run_dir}")
    return SummaryWriter(log_dir=str(run_dir))


def _log_embedding_projector(
    writer: Any,
    model: EegCLAPEncoder,
    val_loader: DataLoader[PairBatch],
    device: torch.device,
) -> None:
    """Snapshot the val embedding space for the TensorBoard Projector tab.

    Metadata labels each point with `trial=T/subj=S` so the UI coloring
    can show trial-level clustering (expected signal) and subject-level
    clustering (subject bias we'd want to minimize).
    """
    eeg_emb, _audio_emb, trial_ids, subject_ids = _chunked_forward(model, val_loader, device)
    metadata = [f"trial={int(t)}/subj={int(s)}" for t, s in zip(trial_ids, subject_ids, strict=True)]
    writer.add_embedding(eeg_emb, metadata=metadata, tag="val_embeddings")


def train(config: ContrastiveConfig) -> Path:
    _set_seed(config.seed)
    device = _get_device()
    logger.info(f"Device: {device}, config: {config}")

    # Build shared audio cache + stimulus list once, pass into all three
    # dataset instances so we don't re-load the npz cache three times.
    shared_resolved = load_resolved_stimuli()
    shared_audio_cache = build_audio_embedding_cache(shared_resolved)

    # Pick the split from the full 32-subject universe, not from whatever
    # subjects happen to be on disk, so the split is deterministic regardless
    # of partial DEAP downloads.
    universe = list(range(1, 33))
    train_subj, val_subj, test_subj = _split_subjects(universe, quick=config.quick, seed=config.seed)
    logger.info(f"Split: train={train_subj} val={val_subj} test={test_subj}")

    # Separate dataset instances (not Subset views) so future augmentation
    # can be gated on `augment=True` without leaking into val/test.
    train_ds = DeapClapPairDataset(
        subject_filter=train_subj,
        augment=True,
        resolved_stimuli=shared_resolved,
        audio_embeddings=shared_audio_cache,
    )
    val_ds = DeapClapPairDataset(
        subject_filter=val_subj,
        augment=False,
        resolved_stimuli=shared_resolved,
        audio_embeddings=shared_audio_cache,
    )
    test_ds = DeapClapPairDataset(
        subject_filter=test_subj,
        augment=False,
        resolved_stimuli=shared_resolved,
        audio_embeddings=shared_audio_cache,
    )

    train_loader: DataLoader[PairBatch] = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader: DataLoader[PairBatch] = DataLoader(val_ds, batch_size=config.batch_size, num_workers=0)
    test_loader: DataLoader[PairBatch] = DataLoader(test_ds, batch_size=config.batch_size, num_workers=0)

    model = EegCLAPEncoder().to(device)
    temperature = torch.nn.Parameter(torch.tensor(math.log(1.0 / 0.07), device=device))

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": config.backbone_lr},
            {"params": model.projection_parameters(), "lr": config.projection_lr},
            {"params": [temperature], "lr": config.projection_lr},
        ],
        weight_decay=config.weight_decay,
    )

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=max(1, config.warmup_epochs),
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.epochs - config.warmup_epochs),
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[max(1, config.warmup_epochs)],
    )

    early = EarlyStopping(patience=config.patience, min_epochs=max(5, config.warmup_epochs + 2))
    best_state: dict[str, torch.Tensor] | None = None
    best_temperature: float = 0.0
    best_val: dict[str, float] = {"loss": float("inf"), "top1": 0.0, "top5": 0.0, "top10": 0.0, "mrr": 0.0}
    best_epoch = 0

    writer = _make_writer(config)
    tb_run_dir = Path(writer.log_dir) if writer is not None else None

    try:
        for epoch in range(1, config.epochs + 1):
            epoch_start = time.monotonic()
            train_loss, timing = _train_one_epoch(
                model=model,
                temperature=temperature,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                grad_accum=config.grad_accum,
            )

            val_start = time.monotonic()
            val_metrics = _evaluate(model, val_loader, temperature=temperature, device=device)
            timing.val_s = time.monotonic() - val_start

            scheduler.step()
            epoch_s = time.monotonic() - epoch_start

            backbone_lr = optimizer.param_groups[0]["lr"]
            projection_lr = optimizer.param_groups[1]["lr"]
            temperature_value = float(temperature.detach().cpu().item())

            logger.info(
                f"Epoch {epoch:3d}/{config.epochs}  "
                f"train_loss={train_loss:.4f}  val {_format_metrics(val_metrics)}  "
                f"lr(b/p)={backbone_lr:.2e}/{projection_lr:.2e}  T={math.exp(temperature_value):.3f}  "
                f"({epoch_s:.1f}s)"
            )
            logger.info(
                f"  Timing: data={timing.data_s:.1f}s ({timing.data_pct(epoch_s):.0f}%, "
                f"{timing.samples_per_second():.0f} samp/s) | "
                f"fwd={timing.forward_s:.1f}s | bwd={timing.backward_s:.1f}s | val={timing.val_s:.1f}s"
            )

            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
                writer.add_scalar("Retrieval/top1_val", val_metrics["top1"], epoch)
                writer.add_scalar("Retrieval/top5_val", val_metrics["top5"], epoch)
                writer.add_scalar("Retrieval/top10_val", val_metrics["top10"], epoch)
                writer.add_scalar("Retrieval/mrr_val", val_metrics["mrr"], epoch)
                writer.add_scalar("LR/backbone", backbone_lr, epoch)
                writer.add_scalar("LR/projection", projection_lr, epoch)
                writer.add_scalar("Temperature", math.exp(temperature_value), epoch)
                writer.add_scalar("Timing/data_pct", timing.data_pct(epoch_s), epoch)
                writer.add_scalar("Timing/samples_per_sec", timing.samples_per_second(), epoch)

            should_stop, is_new_best = early.step(val_metrics["top5"])
            if is_new_best:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_temperature = temperature_value
                best_val = val_metrics
                best_epoch = epoch
            if should_stop:
                logger.info(f"Early stop at epoch {epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        test_metrics = _evaluate(model, test_loader, temperature=temperature, device=device)
        logger.info(f"Test {_format_metrics(test_metrics)}")

        if writer is not None:
            _log_embedding_projector(writer, model, val_loader, device)
    finally:
        if writer is not None:
            writer.close()

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
            "epoch": best_epoch,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "test_size": len(test_ds),
            "git_commit": _git_commit_hash(),
            "tensorboard_run": str(tb_run_dir) if tb_run_dir is not None else None,
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
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=DEFAULT_GRAD_ACCUM,
        help=(
            "Gradient accumulation steps. Simulates batch_size*N effective batch for gradient "
            "smoothness. Note: does NOT grow the in-batch contrastive negative pool."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = ContrastiveConfig(
        seed=args.seed,
        quick=args.quick,
        grad_accum=args.grad_accum,
        use_tensorboard=not args.no_tensorboard,
    )
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
