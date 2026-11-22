"""The quadrant-classifier checkpoint the app loads once at startup, held for the tools that describe it.

A module-level slot rather than ``app.state``: the reader is an agent capability, which
sees ``RunContext`` and never the request. Only ``get_model_info`` and the capability's
rule that hides it read the slot. The checkpoint is optional: without one the app boots
with the slot empty and the assistant simply does not offer ``get_model_info``.
``seed-sessions`` loads its own copy of the checkpoint and does not use this slot.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI

from cortexdj.ml.checkpoint import CHECKPOINT_PATH, QuadrantCheckpoint, load_checkpoint

logger = logging.getLogger(__name__)

_checkpoint: QuadrantCheckpoint | None = None


def get_eeg_model() -> QuadrantCheckpoint | None:
    """The checkpoint loaded at startup, or None when none was found."""
    return _checkpoint


@contextmanager
def override_eeg_model(checkpoint: QuadrantCheckpoint | None) -> Iterator[None]:
    """Swap the held checkpoint for the length of a ``with`` block, then put the previous one back.

    For tests: a checkpoint to describe, or ``None`` to stand for "no checkpoint". The app
    loads once at startup and never calls this.
    """
    global _checkpoint
    previous = _checkpoint
    _checkpoint = checkpoint
    try:
        yield
    finally:
        _checkpoint = previous


def load_eeg_model(checkpoint_path: Path | None = None) -> None:
    """Load the quadrant-classifier checkpoint into the slot; a missing file leaves it empty.

    ``checkpoint_path`` defaults to the file ``train-classifier`` writes; a test passes a path
    of its own. Only a missing file is tolerated: a checkpoint made with a different training
    recipe raises, so the app refuses to start rather than describe a model it would read
    wrongly. Do not soften that to a warning.
    """
    global _checkpoint
    path = checkpoint_path or CHECKPOINT_PATH
    try:
        _checkpoint = load_checkpoint(path)
    except FileNotFoundError:
        logger.warning(
            f"No quadrant-classifier checkpoint at {path}; starting without one. "
            "Run `uv run --directory backend train-classifier` to make it."
        )
        _checkpoint = None
        return
    logger.info(f"Quadrant-classifier checkpoint loaded: {len(_checkpoint.participants)} participants")


async def load_eeg_model_at_startup(app: FastAPI) -> None:
    """The startup hook ``api.py`` registers."""
    load_eeg_model()
