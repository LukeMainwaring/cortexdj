"""The EEG model holder: what startup loads into the slot, and how a test swaps it."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

import cortexdj.api  # noqa: F401  (registers the sessions startup hook)
from cortexdj.core.config import get_settings
from cortexdj.core.startup import get_startup_hooks
from cortexdj.features.sessions.eeg_model import (
    get_eeg_model,
    load_eeg_model,
    load_eeg_model_at_startup,
    override_eeg_model,
)
from cortexdj.ml.model import EEGNetClassifier
from cortexdj.ml.predict import CHECKPOINT_SCHEMA_VERSION


@pytest.fixture
def empty_slot() -> Iterator[None]:
    """Run the test with no model held, and put back whatever was held before, whatever it loads."""
    with override_eeg_model(None):
        yield


@pytest.fixture
def eegnet_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # The default backend is CBraMod, whose loader fetches the pretrained backbone.
    monkeypatch.setenv("EEG_MODEL_BACKEND", "eegnet")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _write_eegnet_checkpoint(path: Path) -> EEGNetClassifier:
    model = EEGNetClassifier()
    torch.save(
        {
            "model_type": "eegnet",
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return model


def test_override_swaps_the_model_and_restores_the_previous_one(empty_slot: None) -> None:
    model = EEGNetClassifier()

    with override_eeg_model(model):
        assert get_eeg_model() is model

    assert get_eeg_model() is None


def test_a_missing_checkpoint_leaves_the_slot_empty_and_names_the_download_script(
    empty_slot: None, eegnet_backend: None, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with override_eeg_model(EEGNetClassifier()):
        with caplog.at_level(logging.WARNING, logger="cortexdj.features.sessions.eeg_model"):
            load_eeg_model(checkpoint_path=tmp_path / "missing.pt")

        assert get_eeg_model() is None

    assert "download-checkpoints.sh" in caplog.text


def test_loads_a_checkpoint_into_the_slot(empty_slot: None, eegnet_backend: None, tmp_path: Path) -> None:
    saved = _write_eegnet_checkpoint(tmp_path / "eegnet_best.pt")

    load_eeg_model(checkpoint_path=tmp_path / "eegnet_best.pt")

    loaded = get_eeg_model()
    assert isinstance(loaded, EEGNetClassifier)
    assert not loaded.training
    for name, tensor in saved.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor)


def test_the_loader_is_a_registered_startup_hook() -> None:
    assert load_eeg_model_at_startup in get_startup_hooks()
