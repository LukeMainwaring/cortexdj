"""Autoresearch: run one training experiment on Modal, log the result.

See ``backend/autoresearch/README.md`` for the full workflow. This script
is intentionally separate from ``modal_train.py`` because the two flows
have different return contracts: ``modal_train.py`` returns checkpoints
for consumption by the inference pipeline, autoresearch returns a scalar
metric + artifacts for an append-only experiment log.

One invocation = one experiment. The agent's loop is: edit
``backend/autoresearch/train.py`` → ``modal run backend/scripts/modal_autoresearch.py``
→ read ``experiments/experiments.jsonl`` → decide keep-or-revert → repeat.

All paths and image settings mirror ``modal_train.py`` so cold-start is
warmed by the same uv/hatch build cache when both apps run in the same
Modal workspace.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
AUTORESEARCH_DIR = BACKEND_DIR / "autoresearch"
EXPERIMENTS_DIR = AUTORESEARCH_DIR / "experiments"
JSONL_PATH = EXPERIMENTS_DIR / "experiments.jsonl"
BEST_PATH = EXPERIMENTS_DIR / "best.json"
RUNS_DIR = EXPERIMENTS_DIR / "runs"

REMOTE_APP_DIR = "/app"
REMOTE_BACKEND = f"{REMOTE_APP_DIR}/backend"
REMOTE_DEAP = f"{REMOTE_BACKEND}/data/deap"
REMOTE_AUTORESEARCH = f"{REMOTE_BACKEND}/autoresearch"

DEAP_VOLUME_NAME = "cortexdj-deap"

app = modal.App("cortexdj-autoresearch")

# Mirrors modal_train.py exclusions. `autoresearch/experiments/**` is added
# here so the ever-growing run log doesn't bloat cold-start uploads.
BACKEND_IGNORE = [
    "**/.venv",
    "**/__pycache__",
    "**/*.pyc",
    "**/.mypy_cache",
    "**/.ruff_cache",
    "**/.pytest_cache",
    "**/.DS_Store",
    "data/deap/**",
    "data/checkpoints/**",
    "data/synthetic/**",
    "data/tensorboard_runs/**",
    "data/track_index_miss_log.jsonl",
    "data/deap_stimuli_miss_log.jsonl",
    "autoresearch/experiments/**",
]

image = (
    modal.Image.from_registry(
        "ghcr.io/astral-sh/uv:python3.13-bookworm-slim",
        add_python=None,
    )
    .env({"UV_COMPILE_BYTECODE": "1", "UV_LINK_MODE": "copy"})
    .apt_install("ffmpeg")
    .add_local_dir(str(BACKEND_DIR), REMOTE_BACKEND, copy=True, ignore=BACKEND_IGNORE)
    .add_local_file(str(REPO_ROOT / "README.md"), f"{REMOTE_APP_DIR}/README.md", copy=True)
    .run_commands(f"uv sync --directory {REMOTE_BACKEND} --frozen --no-dev")
)

deap_volume = modal.Volume.from_name(DEAP_VOLUME_NAME)


def _now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


@app.cls(
    image=image,
    gpu="A10G",
    volumes={REMOTE_DEAP: deap_volume},
    # 15-min training + ~15 min cold-start + slack. No retries: an
    # autoresearch run failing is a legitimate signal the agent needs to see.
    timeout=3600,
)
class Researcher:
    @modal.method()
    def run(self, run_id: str) -> dict[str, Any]:
        """Run one autoresearch experiment; return artifacts + the scalar metric."""
        run_dir = Path(f"/tmp/autoresearch_runs/{run_id}")
        run_dir.mkdir(parents=True, exist_ok=True)

        train_py_path = Path(f"{REMOTE_AUTORESEARCH}/train.py")
        train_py_bytes = train_py_path.read_bytes()
        (run_dir / "train.py").write_bytes(train_py_bytes)

        env = {**os.environ, "AUTORESEARCH_RUN_DIR": str(run_dir)}
        cmd = ["uv", "run", "--directory", REMOTE_BACKEND, "python", "-m", "autoresearch.train"]

        stdout_lines: list[str] = []
        start = time.monotonic()
        with subprocess.Popen(
            cmd,
            cwd=REMOTE_BACKEND,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as p:
            assert p.stdout is not None
            for line in p.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                stdout_lines.append(line)
            p.wait()
            returncode = p.returncode

        duration_s = round(time.monotonic() - start, 2)
        stdout_text = "".join(stdout_lines)
        (run_dir / "stdout.log").write_text(stdout_text)

        metrics_path = run_dir / "metrics.json"
        metrics: dict[str, Any] = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
            except Exception as e:
                print(f"[autoresearch] failed to parse metrics.json: {e}")

        metric = _pick_metric(metrics, stdout_text)
        status = _classify_status(returncode, metric)

        # Preprocessing cache (.npz) may have been regenerated during this run;
        # commit so the next experiment doesn't pay the cost again.
        deap_volume.commit()

        return {
            "run_id": run_id,
            "status": status,
            "metric": metric,
            "metrics": metrics,
            "duration_s": duration_s,
            "returncode": returncode,
            "stdout": stdout_text,
            "train_py": train_py_bytes,
            "metrics_json": metrics_path.read_bytes() if metrics_path.exists() else b"",
        }


def _pick_metric(metrics: dict[str, Any], stdout_text: str) -> float | None:
    """Prefer metrics.json; fall back to grepping FINAL_METRIC from stdout.

    NaN / +-inf are treated as "no metric" — letting them through would
    poison best.json (any ``metric > NaN`` is False, so the champion
    locks forever). Non-finite values typically come from a crashed or
    diverged run that the agent should treat as a failure anyway.
    """
    for key in ("best_avg_macro_f1", "avg_macro_f1"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)

    final: float | None = None
    for line in stdout_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("FINAL_METRIC="):
            try:
                parsed = float(stripped.split("=", 1)[1])
            except ValueError:
                continue
            if math.isfinite(parsed):
                final = parsed
    return final


def _classify_status(returncode: int, metric: float | None) -> str:
    if returncode != 0:
        return "failed"
    if metric is None:
        return "no_metric"
    return "ok"


@app.local_entrypoint()
def main(gpu: str = "A10G") -> None:
    """Launch one experiment on Modal, persist artifacts, append a JSONL entry."""
    run_id = _now_run_id()
    print(f"[autoresearch] launching run {run_id} on {gpu}")

    local_run_dir = RUNS_DIR / run_id
    local_run_dir.mkdir(parents=True, exist_ok=True)

    train_py_local = (AUTORESEARCH_DIR / "train.py").read_bytes()
    train_py_sha = hashlib.sha256(train_py_local).hexdigest()

    runner = Researcher.with_options(gpu=gpu)()  # type: ignore[attr-defined]
    try:
        result = cast(dict[str, Any], runner.run.remote(run_id))
    except Exception as exc:
        # Modal failed before returning — auth, timeout, worker death, network.
        # The agent's loop tails experiments.jsonl to decide keep-or-revert;
        # if we don't log here, the loop silently breaks its invariant. Log an
        # infra_failed row and re-raise so the caller also sees the error.
        _append_infra_failure(run_id, gpu, train_py_sha, repr(exc))
        raise

    train_py_returned = result.get("train_py") or train_py_local
    (local_run_dir / "train.py").write_bytes(bytes(train_py_returned))
    stdout_text = str(result.get("stdout", ""))
    (local_run_dir / "stdout.log").write_text(stdout_text)
    metrics_bytes = bytes(result.get("metrics_json") or b"")
    if metrics_bytes:
        (local_run_dir / "metrics.json").write_bytes(metrics_bytes)

    metric = result.get("metric")
    metric_f = float(metric) if isinstance(metric, (int, float)) else None
    status = str(result.get("status", "unknown"))
    metrics_dict = result.get("metrics") or {}
    if not isinstance(metrics_dict, dict):
        metrics_dict = {}

    is_best = metric_f is not None and _update_best(local_run_dir, run_id, metric_f)

    log_entry: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "metric": metric_f,
        "arousal_f1": metrics_dict.get("arousal_f1"),
        "valence_f1": metrics_dict.get("valence_f1"),
        "duration_s": result.get("duration_s"),
        "steps": metrics_dict.get("steps"),
        "epochs": metrics_dict.get("epochs"),
        "train_py_sha256": train_py_sha,
        "is_best": is_best,
        "gpu": gpu,
    }

    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("a") as f:
        f.write(json.dumps(log_entry) + "\n")

    _print_summary(log_entry)


def _append_infra_failure(run_id: str, gpu: str, train_py_sha: str, reason: str) -> None:
    """Log a JSONL row when the Modal call itself fails (as opposed to training failing).

    Keeps the agent's ``tail experiments.jsonl`` loop invariant: every
    invocation produces exactly one log line.
    """
    entry: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "infra_failed",
        "metric": None,
        "arousal_f1": None,
        "valence_f1": None,
        "duration_s": None,
        "steps": None,
        "epochs": None,
        "train_py_sha256": train_py_sha,
        "is_best": False,
        "gpu": gpu,
        "error": reason,
    }
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _update_best(run_dir: Path, run_id: str, metric: float) -> bool:
    # Defense in depth — _pick_metric already filters non-finite values, but a
    # bad caller slipping NaN through would permanently lock best.json.
    if not math.isfinite(metric):
        return False
    current_best: float | None = None
    if BEST_PATH.exists():
        try:
            current_best = float(json.loads(BEST_PATH.read_text()).get("metric", 0.0))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[autoresearch] WARN: best.json unreadable ({exc!r}); overwriting with {metric:.4f}")
            current_best = None
    if current_best is None or metric > current_best:
        BEST_PATH.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "metric": metric,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "train_py_path": str(run_dir / "train.py"),
                },
                indent=2,
            )
        )
        return True
    return False


def _print_summary(entry: dict[str, Any]) -> None:
    metric = entry["metric"]
    best_tag = " [BEST]" if entry.get("is_best") else ""
    metric_str = f"{metric:.4f}" if isinstance(metric, (int, float)) else "<none>"
    print(
        f"\n[autoresearch] run={entry['run_id']} "
        f"status={entry['status']} "
        f"metric={metric_str}{best_tag} "
        f"duration={entry.get('duration_s')}s"
    )
