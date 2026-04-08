"""Canonical data paths — single source of truth for all data directory references."""

from pathlib import Path

# 4 parents: core/ → cortexdj/ → src/ → backend root (host) or /app/ (Docker)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATA_DIR = _BACKEND_ROOT / "data"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
DEAP_DATA_DIR = DATA_DIR / "deap"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
