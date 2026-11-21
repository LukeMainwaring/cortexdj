"""The contrastive encoder holder: what startup loads into the slot, and how a test swaps it.

Loading a real checkpoint is not tested: ``load_contrastive_encoder`` builds the encoder with
the pretrained CBraMod backbone, which downloads from the Hugging Face hub.
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from cortexdj.features.retrieval.encoder import (
    ContrastiveEncoder,
    get_contrastive_encoder,
    load_contrastive_encoder,
    override_contrastive_encoder,
)

# These tests only compare identity, so any object stands in for a loaded encoder.
_ENCODER = cast(ContrastiveEncoder, object())


@pytest.fixture
def empty_slot() -> Iterator[None]:
    """Run the test with no encoder held, and put back whatever was held before, whatever it loads."""
    with override_contrastive_encoder(None):
        yield


def test_override_swaps_the_encoder_and_restores_the_previous_one(empty_slot: None) -> None:
    with override_contrastive_encoder(_ENCODER):
        assert get_contrastive_encoder() is _ENCODER
        with override_contrastive_encoder(None):
            assert get_contrastive_encoder() is None
        assert get_contrastive_encoder() is _ENCODER

    assert get_contrastive_encoder() is None


def test_a_missing_checkpoint_leaves_the_slot_empty_and_names_the_download_script(
    empty_slot: None, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with override_contrastive_encoder(_ENCODER):
        with caplog.at_level(logging.WARNING, logger="cortexdj.features.retrieval.encoder"):
            load_contrastive_encoder(tmp_path / "missing.pt")

        assert get_contrastive_encoder() is None

    assert "download-checkpoints.sh" in caplog.text
