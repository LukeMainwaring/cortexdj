"""GPU training on Modal — run CortexDJ model training on a cloud GPU.

Setup (one-time):
    pip install modal
    modal setup          # authenticates via browser

Usage:
    # Train CBraMod with best-effort defaults on A10G (~$1-2 for full LOSO)
    modal run backend/scripts/modal_train.py

    # Quick test run
    modal run backend/scripts/modal_train.py -- --quick

    # Train EEGNet instead
    modal run backend/scripts/modal_train.py -- --model eegnet

    # Compare both models
    modal run backend/scripts/modal_train.py --command compare-models

    # Use a different GPU
    modal run backend/scripts/modal_train.py --gpu a100

Checkpoints are saved locally to backend/data/checkpoints/ when the run completes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

# ── Paths (relative to repo root) ───────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
DEAP_DIR = BACKEND_DIR / "data" / "deap"
CHECKPOINTS_DIR = BACKEND_DIR / "data" / "checkpoints"

REMOTE_APP_DIR = "/app"
REMOTE_BACKEND = f"{REMOTE_APP_DIR}/backend"
REMOTE_DEAP = f"{REMOTE_BACKEND}/data/deap"
REMOTE_CHECKPOINTS = f"{REMOTE_BACKEND}/data/checkpoints"

# ── Modal setup ──────────────────────────────────────────────────────────────
app = modal.App("cortexdj-train")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .pip_install("uv")
    .copy_local_dir(str(BACKEND_DIR), f"{REMOTE_BACKEND}")
    .run_commands(f"uv sync --directory {REMOTE_BACKEND}")
)

deap_mount = modal.Mount.from_local_dir(str(DEAP_DIR), remote_path=REMOTE_DEAP)


@app.function(
    image=image,
    gpu="a10g",
    mounts=[deap_mount],
    timeout=7200,  # 2 hours
)
def train(args: list[str], command: str = "train-model") -> dict[str, bytes]:
    """Run training on a remote GPU and return checkpoint files."""
    cmd = ["uv", "run", "--directory", REMOTE_BACKEND, command, *args]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Collect all checkpoint files to return
    checkpoint_dir = Path(REMOTE_CHECKPOINTS)
    checkpoints: dict[str, bytes] = {}
    if checkpoint_dir.exists():
        for pt_file in checkpoint_dir.glob("*.pt"):
            checkpoints[pt_file.name] = pt_file.read_bytes()

    if not checkpoints:
        print("Warning: no checkpoint files found after training")
    return checkpoints


@app.local_entrypoint()
def main(command: str = "train-model", gpu: str = "a10g"):
    """Local entrypoint — pass training args after `--`.

    Args:
        command: train-model or compare-models
        gpu: Modal GPU type (t4, a10g, a100, h100)
    """
    import sys

    # Everything after `--` is passed through to train-model
    train_args: list[str] = []
    if "--" in sys.argv:
        sep_idx = sys.argv.index("--")
        train_args = sys.argv[sep_idx + 1 :]

    # Override GPU if specified (requires recreating the function with new GPU)
    train_fn = train
    if gpu != "a10g":
        train_fn = train.with_options(gpu=gpu)

    print(f"Starting {command} on Modal ({gpu})...")
    checkpoints = train_fn.remote(train_args, command=command)

    # Save checkpoints locally
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in checkpoints.items():
        path = CHECKPOINTS_DIR / name
        path.write_bytes(data)
        print(f"Saved {path} ({len(data) / 1024 / 1024:.1f} MB)")

    if checkpoints:
        print(f"\nDone — {len(checkpoints)} checkpoint(s) saved to {CHECKPOINTS_DIR}")
    else:
        print("\nTraining completed but no checkpoints were produced")
