"""Thin wrapper: ``uv run --directory backend python scripts/run_autoresearch.py``.

Exists so the agent's loop has a stable, short invocation surface that
doesn't hardcode ``modal run`` in its instructions. If we later swap
Modal for another backend, only this file has to change.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODAL_SCRIPT = Path(__file__).resolve().parent / "modal_autoresearch.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one autoresearch experiment on Modal.")
    parser.add_argument("--gpu", default="A10G", help="Modal GPU type (T4, A10G, A100, H100)")
    args = parser.parse_args()

    cmd = ["modal", "run", str(MODAL_SCRIPT), "--gpu", args.gpu]
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    sys.exit(main())
