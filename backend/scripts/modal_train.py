"""GPU training on Modal — run CortexDJ model training on a cloud GPU.

Setup (one-time):
    pip install modal
    modal setup          # authenticates via browser

Usage:
    # Train CBraMod with production defaults on A10G (~$1-2 for full LOSO)
    modal run backend/scripts/modal_train.py

    # Quick test run
    modal run backend/scripts/modal_train.py -- --quick

    # Train EEGNet instead
    modal run backend/scripts/modal_train.py -- --model eegnet

    # Compare both models
    modal run backend/scripts/modal_train.py --command compare-models

    # Use a different GPU (t4 ~$0.50/hr, a10g ~$1/hr, a100 ~$3/hr)
    modal run backend/scripts/modal_train.py --gpu a100

Checkpoints are saved locally to backend/data/checkpoints/ when the run completes.
Requires DEAP data in backend/data/deap/ (see backend/data/DEAP_SETUP.md).
"""

from __future__ import annotations

import subprocess
import sys
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

VALID_COMMANDS = {"train-model", "compare-models"}

# ── Validate DEAP data exists before Modal tries to mount it ─────────────────
if not DEAP_DIR.exists():
    print(
        f"Error: DEAP data directory not found at {DEAP_DIR}\n"
        "See backend/data/DEAP_SETUP.md for download instructions.",
        file=sys.stderr,
    )
    raise SystemExit(1)

# ── Modal setup ──────────────────────────────────────────────────────────────
# Image excludes .venv, data/deap, caches via backend/.modalignore
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
    timeout=14400,  # 4 hours (T4 may need this for full LOSO)
)
def train(args: list[str], command: str = "train-model") -> dict[str, bytes]:
    """Run training on a remote GPU and return checkpoint files."""
    if command not in VALID_COMMANDS:
        msg = f"Unknown command: {command!r}. Must be one of {VALID_COMMANDS}"
        raise ValueError(msg)

    cmd = ["uv", "run", "--directory", REMOTE_BACKEND, command, *args]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Collect checkpoint files to return
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
    # Everything after `--` is passed through to train-model
    train_args: list[str] = []
    if "--" in sys.argv:
        sep_idx = sys.argv.index("--")
        train_args = sys.argv[sep_idx + 1 :]

    # Override GPU if specified
    run_fn = train.with_options(gpu=gpu) if gpu != "a10g" else train

    print(f"Starting {command} on Modal ({gpu})...")
    checkpoints = run_fn.remote(train_args, command=command)

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
