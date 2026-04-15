"""GPU training on Modal — runs `train-model` / `train-contrastive` / `compare-models` on a cloud GPU.

See ``DEVELOPMENT.md`` § GPU Training (Modal) for one-time setup (``modal setup``,
seeding the ``cortexdj-deap`` volume) and the full command reference.

Entrypoint args (see ``main`` below):
    command: ``train-model`` (default), ``train-contrastive``, or ``compare-models``
    gpu:     Modal GPU type — ``T4``, ``A10G`` (default), ``A100``, ``H100``
    args:    quoted passthrough string for the underlying console script,
             e.g. ``--args="--quick --model eegnet"``

Contrastive training notes:
    - Ships the hand-curated ``backend/data/deap_stimuli.json`` + resolved
      sidecar + m4a audio cache with the image (not via the Modal volume)
      so the worker has everything needed to rebuild the CLAP audio cache
      without re-fetching from iTunes.
    - Disable tensorboard inside the container run with
      ``--args="--no-tensorboard"`` if you want to skip writing event files
      to the ephemeral worker disk. Otherwise the TB run dir is created but
      not currently downloaded back to the local machine.
    - The ``contrastive_best.pt`` checkpoint is downloaded the same way
      ``cbramod_best.pt`` is, via the ``data/checkpoints/`` glob.

Checkpoints are downloaded to ``backend/data/checkpoints/`` when the run completes.
DEAP source data lives in the ``cortexdj-deap`` Modal Volume.

**Long runs and laptop sleep.** The Modal client must stay attached for the
checkpoint bytes to make it back to the local machine. Wrap long training runs
in ``caffeinate -dim`` so macOS can't sleep mid-run and drop the connection::

    caffeinate -dim modal run backend/scripts/modal_train.py

**Preemption safety.** Modal restarts a preempted Function with the same input;
the training loop persists fold-level resume state under ``data/deap/.train_state/``
on the DEAP volume so the restart picks up at the last completed fold instead
of re-running from fold 0. Use ``--args="--no-resume"`` for a clean restart.
"""

import shlex
import subprocess
from pathlib import Path

import modal

# ── Paths (relative to repo root) ───────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
CHECKPOINTS_DIR = BACKEND_DIR / "data" / "checkpoints"

REMOTE_APP_DIR = "/app"
REMOTE_BACKEND = f"{REMOTE_APP_DIR}/backend"
REMOTE_DEAP = f"{REMOTE_BACKEND}/data/deap"
REMOTE_CHECKPOINTS = f"{REMOTE_BACKEND}/data/checkpoints"

VALID_COMMANDS = {"train-model", "train-contrastive", "compare-models"}
DEAP_VOLUME_NAME = "cortexdj-deap"

# ── Modal setup ──────────────────────────────────────────────────────────────
app = modal.App("cortexdj-train")

# Keep in sync with backend/Dockerfile — same base, same uv env vars, same sync flags.
# Excludes match what we don't want baked into the image (caches, venv, large data dirs).
BACKEND_IGNORE = [
    "**/.venv",
    "**/__pycache__",
    "**/*.pyc",
    "**/.mypy_cache",
    "**/.ruff_cache",
    "**/.pytest_cache",
    "**/.DS_Store",
    "data/deap/**",  # lives in the cortexdj-deap Modal Volume
    "data/checkpoints/**",  # training output, not input
    "data/synthetic/**",  # unused by DEAP training
    "data/tensorboard_runs/**",  # training output, not input; keeps image lean
    "data/track_index_miss_log.jsonl",  # runtime log, contains user library refs
    "data/deap_stimuli_miss_log.jsonl",  # runtime log, not needed on worker
]

image = (
    modal.Image.from_registry(
        "ghcr.io/astral-sh/uv:python3.13-bookworm-slim",
        add_python=None,  # image already ships python3.13
    )
    .env({"UV_COMPILE_BYTECODE": "1", "UV_LINK_MODE": "copy"})
    # ffmpeg gives librosa's audioread fallback a decoder for the iTunes m4a
    # (AAC) previews — libsndfile can't read AAC, and the slim base has no
    # system audio backend, so without this contrastive training dies on the
    # first audio load with audioread.NoBackendError.
    .apt_install("ffmpeg")
    .add_local_dir(str(BACKEND_DIR), REMOTE_BACKEND, copy=True, ignore=BACKEND_IGNORE)
    # backend/pyproject.toml has `readme = "../README.md"` — hatchling reads it at
    # build time, so the repo-root README has to land at /app/README.md.
    .add_local_file(str(REPO_ROOT / "README.md"), f"{REMOTE_APP_DIR}/README.md", copy=True)
    .run_commands(f"uv sync --directory {REMOTE_BACKEND} --frozen --no-dev")
)

# DEAP (6.7 GB) lives in a persistent Volume seeded once via `modal volume put`.
# create_if_missing is intentionally False: if the volume doesn't exist, fail fast
# so the user runs the seed step from the module docstring rather than training
# against an empty volume.
deap_volume = modal.Volume.from_name(DEAP_VOLUME_NAME)


# Wrapped in @app.cls so we can use Cls.with_options(gpu=...) to override the GPU
# at call time. Plain @app.function does not support .with_options() in Modal 1.x.
# See docs/modal-llms-full.txt for the canonical pattern.
@app.cls(
    image=image,
    gpu="A10G",
    volumes={REMOTE_DEAP: deap_volume},
    timeout=14400,  # 4 hours (T4 may need this for full LOSO)
    # Modal auto-restarts on preemption; this extends that behavior to other
    # transient failures. Combined with the in-loop resume state, a restart
    # picks up at the last completed fold rather than redoing prior work.
    # backoff_coefficient=1.0 (flat, no exponential) because the expected
    # failure mode is spot-instance preemption — we want the retry to land
    # on a fresh worker as fast as possible, not wait out a backoff window.
    # TODO: contrastive training downloads ~1.9 GB of LAION-CLAP weights via
    # `transformers.ClapModel.from_pretrained` on every cold start. A
    # preempted run re-downloads them; a warm cache would save ~2-5 min per
    # retry. Add a `modal.Volume.from_name("cortexdj-hf-cache")` mounted at
    # `/root/.cache/huggingface` when this becomes a bottleneck.
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0),
)
class Trainer:
    @modal.method()
    def train(self, args: list[str], command: str = "train-model") -> dict[str, bytes]:
        cmd = ["uv", "run", "--directory", REMOTE_BACKEND, command, *args]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Persist any preprocessing cache (data/deap/.cache/*.npz) that dataset.py
        # generated during this run, so subsequent runs reuse it instead of recomputing.
        deap_volume.commit()

        checkpoint_dir = Path(REMOTE_CHECKPOINTS)
        checkpoints: dict[str, bytes] = {}
        if checkpoint_dir.exists():
            for pt_file in checkpoint_dir.glob("*.pt"):
                checkpoints[pt_file.name] = pt_file.read_bytes()

        if not checkpoints:
            print("Warning: no checkpoint files found after training")
        return checkpoints


@app.local_entrypoint()
def main(command: str = "train-model", gpu: str = "A10G", args: str = "") -> None:
    """Local entrypoint.

    Args:
        command: train-model or compare-models
        gpu: Modal GPU type (T4, A10G, A100, H100)
        args: quoted string of passthrough args for the underlying console script,
              e.g. --args="--quick --model eegnet"
    """
    if command not in VALID_COMMANDS:
        msg = f"Unknown command: {command!r}. Must be one of {sorted(VALID_COMMANDS)}"
        raise SystemExit(msg)

    train_args = shlex.split(args)
    # Trainer.with_options(gpu=gpu) returns a new class handle; calling it (`()`)
    # gives an instance whose .train.remote(...) runs on the overridden GPU.
    # mypy doesn't see Modal's @app.cls dynamic transforms; runtime is correct
    # per https://modal.com/docs/reference/modal.Cls#with_options.
    runner = Trainer.with_options(gpu=gpu)()  # type: ignore[attr-defined]

    print(f"Starting {command} on Modal ({gpu})...")
    checkpoints = runner.train.remote(train_args, command)

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
