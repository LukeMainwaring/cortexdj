"""The contrastive EEG↔CLAP encoder the app loads once at startup, held for retrieval.

A module-level slot rather than ``app.state``: one of the readers is an agent tool, which
sees ``RunContext`` and never the request; the route reaches it through a FastAPI
dependency wrapping ``get_contrastive_encoder``. Loading at startup makes boot slower, and
a dev server's reload loads both this encoder and the EEG classifier again. The checkpoint
is optional: without one the app boots with the slot empty and retrieval fails with a
message saying how to fetch it.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from fastapi import FastAPI

from cortexdj.core.paths import CHECKPOINTS_DIR
from cortexdj.ml.contrastive import EegCLAPEncoder
from cortexdj.ml.device import get_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContrastiveEncoder:
    """The loaded encoder and the device its weights live on."""

    model: EegCLAPEncoder
    device: torch.device


_encoder: ContrastiveEncoder | None = None


def get_contrastive_encoder() -> ContrastiveEncoder | None:
    """The encoder loaded at startup, or None when no checkpoint was found."""
    return _encoder


@contextmanager
def override_contrastive_encoder(encoder: ContrastiveEncoder | None) -> Iterator[None]:
    """Swap the held encoder for the length of a ``with`` block, then put the previous one back.

    For tests: an encoder to retrieve with, or ``None`` to stand for "no checkpoint". The
    app loads once at startup and never calls this.
    """
    global _encoder
    previous = _encoder
    _encoder = encoder
    try:
        yield
    finally:
        _encoder = previous


def load_contrastive_encoder(checkpoint_path: Path = CHECKPOINTS_DIR / "contrastive_best.pt") -> None:
    """Load the contrastive checkpoint into the slot; a missing checkpoint leaves it empty.

    A test passes a ``checkpoint_path`` of its own. Only a missing file is tolerated: a
    stale or corrupt checkpoint still fails startup.
    """
    global _encoder
    if not checkpoint_path.exists():
        logger.warning(
            f"No contrastive checkpoint found at {checkpoint_path}; starting without the contrastive encoder. "
            "Run `./backend/scripts/download-checkpoints.sh` to fetch it."
        )
        _encoder = None
        return

    device = get_device()
    # weights_only=False is required because our checkpoint is a dict containing
    # non-tensor metadata (schema_version, config, git_commit, train/val/test
    # subject lists). The checkpoint is a trusted local artifact so the
    # deserialization risk is acceptable.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = EegCLAPEncoder().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    _encoder = ContrastiveEncoder(model=model, device=device)
    logger.info(
        f"Loaded contrastive encoder from {checkpoint_path.name} "
        f"(schema_version={checkpoint.get('schema_version')}, "
        f"git_commit={checkpoint.get('git_commit', 'unknown')}) on {device}"
    )


async def load_contrastive_encoder_at_startup(app: FastAPI) -> None:
    """The startup hook ``api.py`` registers."""
    load_contrastive_encoder()
