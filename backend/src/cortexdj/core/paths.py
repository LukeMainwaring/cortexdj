"""Canonical data paths — single source of truth for all data directory references."""

from pathlib import Path

# 4 parents: core/ → cortexdj/ → src/ → backend root (host) or /app/ (Docker)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATA_DIR = _BACKEND_ROOT / "data"
DEAP_DATA_DIR = DATA_DIR / "deap"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
AUDIO_CACHE_DIR = DATA_DIR / "audio_cache"
TENSORBOARD_RUNS_DIR = DATA_DIR / "tensorboard_runs"
# Fold-level resume state for long LOSO runs. Lives under the DEAP data dir
# so it rides along on the `cortexdj-deap` Modal volume without needing a
# second volume mount.
TRAIN_STATE_DIR = DEAP_DATA_DIR / ".train_state"
