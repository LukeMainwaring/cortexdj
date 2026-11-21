"""The contrastive encoder holder: what startup loads into the slot, and how a test swaps it.

Loading a real checkpoint is not tested: ``load_contrastive_encoder`` builds the encoder with
the pretrained CBraMod backbone, which downloads from the Hugging Face hub.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import cortexdj.api  # noqa: F401  (registers the retrieval startup hook)
from cortexdj.core.startup import get_startup_hooks
from cortexdj.features.retrieval.encoder import (
    ContrastiveEncoder,
    get_contrastive_encoder,
    load_contrastive_encoder,
    load_contrastive_encoder_at_startup,
    override_contrastive_encoder,
)
from cortexdj.ml.contrastive import EegCLAPEncoder


class _FakeBackbone(nn.Module):
    """CBraMod's ``forward(x, return_features=True) -> {"features": (B, patches, dim)}`` contract, tiny."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(32 * 800, 8)

    def forward(self, x: torch.Tensor, *, return_features: bool = False) -> dict[str, torch.Tensor]:
        return {"features": self.linear(x.reshape(x.shape[0], -1)).reshape(x.shape[0], 1, 8)}


def _encoder() -> ContrastiveEncoder:
    return ContrastiveEncoder(model=EegCLAPEncoder(backbone=_FakeBackbone()), device=torch.device("cpu"))


@pytest.fixture
def empty_slot() -> Iterator[None]:
    """Run the test with no encoder held, and put back whatever was held before, whatever it loads."""
    with override_contrastive_encoder(None):
        yield


def test_the_slot_is_readable_before_anything_loads_into_it() -> None:
    held = get_contrastive_encoder()

    assert held is None or isinstance(held, ContrastiveEncoder)


def test_override_swaps_the_encoder_and_restores_the_previous_one(empty_slot: None) -> None:
    encoder = _encoder()

    with override_contrastive_encoder(encoder):
        assert get_contrastive_encoder() is encoder
        with override_contrastive_encoder(None):
            assert get_contrastive_encoder() is None
        assert get_contrastive_encoder() is encoder

    assert get_contrastive_encoder() is None


def test_a_missing_checkpoint_leaves_the_slot_empty_and_names_the_download_script(
    empty_slot: None, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with override_contrastive_encoder(_encoder()):
        with caplog.at_level(logging.WARNING, logger="cortexdj.features.retrieval.encoder"):
            load_contrastive_encoder(tmp_path / "missing.pt")

        assert get_contrastive_encoder() is None

    assert "download-checkpoints.sh" in caplog.text


def test_the_loader_is_a_registered_startup_hook() -> None:
    assert load_contrastive_encoder_at_startup in get_startup_hooks()
