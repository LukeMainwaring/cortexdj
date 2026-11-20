"""The EEG classifier the app loads once at startup, held for the tools that describe it.

A module-level slot rather than ``app.state``: the reader is an agent capability, which
sees ``RunContext`` and never the request. Only ``get_model_info`` and the capability's
rule that hides it read the slot. The checkpoint is optional: without one the app boots
with the slot empty and the assistant simply does not offer ``get_model_info``.
``seed-sessions`` loads its own model and does not use this slot.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI

from cortexdj.core.config import get_settings
from cortexdj.ml.predict import EEGModel, load_model

logger = logging.getLogger(__name__)

_model: EEGModel | None = None


def get_eeg_model() -> EEGModel | None:
    """The model loaded at startup, or None when no checkpoint was found."""
    return _model


@contextmanager
def override_eeg_model(model: EEGModel | None) -> Iterator[None]:
    """Swap the held model for the length of a ``with`` block, then put the previous one back.

    For tests: a model to describe, or ``None`` to stand for "no checkpoint". The app
    loads once at startup and never calls this.
    """
    global _model
    previous = _model
    _model = model
    try:
        yield
    finally:
        _model = previous


def load_eeg_model(checkpoint_path: Path | None = None) -> None:
    """Load the ``EEG_MODEL_BACKEND`` checkpoint into the slot; a missing checkpoint leaves it empty.

    ``checkpoint_path`` defaults to that backend's standard checkpoint; a test passes a path
    of its own. Only a missing file is tolerated: a stale or corrupt checkpoint still fails
    startup, with ``load_model``'s message saying how to replace it.
    """
    global _model
    model_type = get_settings().EEG_MODEL_BACKEND
    try:
        _model = load_model(checkpoint_path=checkpoint_path, model_type=model_type)
    except FileNotFoundError:
        logger.warning(
            f"No {model_type} checkpoint found; starting without an EEG model. "
            "Run `./backend/scripts/download-checkpoints.sh` to fetch it."
        )
        _model = None
        return
    logger.info(f"{model_type} model loaded successfully")


async def load_eeg_model_at_startup(app: FastAPI) -> None:
    """The startup hook ``api.py`` registers."""
    load_eeg_model()
