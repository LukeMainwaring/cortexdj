"""The EEG model holder: what startup loads into the slot, and how a test swaps it."""

import logging
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

import cortexdj.api  # noqa: F401  (registers the sessions startup hook)
from cortexdj.core.startup import get_startup_hooks
from cortexdj.features.sessions.eeg_model import (
    get_eeg_model,
    load_eeg_model,
    load_eeg_model_at_startup,
    override_eeg_model,
)
from cortexdj.ml.checkpoint import save_checkpoint
from tests.factories import make_quadrant_checkpoint


@pytest.fixture
def empty_slot() -> Iterator[None]:
    """Run the test with no checkpoint held, and put back whatever was held before, whatever it loads."""
    with override_eeg_model(None):
        yield


def test_override_swaps_the_checkpoint_and_restores_the_previous_one(empty_slot: None) -> None:
    checkpoint = make_quadrant_checkpoint()

    with override_eeg_model(checkpoint):
        assert get_eeg_model() is checkpoint

    assert get_eeg_model() is None


def test_a_missing_checkpoint_leaves_the_slot_empty_and_names_train_classifier(
    empty_slot: None, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with override_eeg_model(make_quadrant_checkpoint()):
        with caplog.at_level(logging.WARNING, logger="cortexdj.features.sessions.eeg_model"):
            load_eeg_model(checkpoint_path=tmp_path / "missing.pt")

        assert get_eeg_model() is None

    assert "train-classifier" in caplog.text


def test_loads_the_checkpoint_with_its_recipe_and_metrics(empty_slot: None, tmp_path: Path) -> None:
    saved = make_quadrant_checkpoint((1, 2))
    save_checkpoint(saved, tmp_path / "quadrant_classifier.pt")

    load_eeg_model(checkpoint_path=tmp_path / "quadrant_classifier.pt")

    loaded = get_eeg_model()
    assert loaded is not None
    assert loaded.recipe == saved.recipe
    assert loaded.summary == saved.summary
    assert sorted(loaded.participants) == [1, 2]


def test_a_recipe_mismatch_fails_the_load_instead_of_warning(empty_slot: None, tmp_path: Path) -> None:
    stale = make_quadrant_checkpoint()
    path = tmp_path / "quadrant_classifier.pt"
    save_checkpoint(replace(stale, recipe=replace(stale.recipe, epochs=stale.recipe.epochs + 1)), path)

    with pytest.raises(ValueError, match="epochs"):
        load_eeg_model(checkpoint_path=path)

    assert get_eeg_model() is None


def test_the_loader_is_a_registered_startup_hook() -> None:
    assert load_eeg_model_at_startup in get_startup_hooks()
