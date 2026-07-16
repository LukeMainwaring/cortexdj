"""Round-trip tests for the fold-level resume state used by long LOSO runs.

The resume path is what lets a preempted Modal run pick up at the last
completed fold instead of re-running from fold 0. These tests exercise
the three failure modes that would break that contract: a clean write→read
round-trip, a stale `run_key` invalidating the file, and an absent file
returning `None`.
"""

from pathlib import Path

import pytest
import torch

from cortexdj.ml import train as train_mod
from cortexdj.ml.train import (
    TrainingConfig,
    _clear_resume_state,
    _load_resume_state,
    _resume_paths,
    _run_key,
    _write_resume_state,
)


@pytest.fixture
def tmp_train_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect TRAIN_STATE_DIR to a tmp_path for the duration of a test.

    The helpers read `train_mod.TRAIN_STATE_DIR` at call time, so monkey-
    patching the module attribute is enough — no need to rebuild paths.
    """
    monkeypatch.setattr(train_mod, "TRAIN_STATE_DIR", tmp_path)
    return tmp_path


def _sample_metrics(f1: float) -> dict[str, object]:
    """Minimal fold-metrics dict mirroring `_evaluate`'s return shape."""
    return {
        "avg_macro_f1": f1,
        "arousal_macro_f1": f1,
        "valence_macro_f1": f1,
        "arousal_recall": [0.5, 0.5],
        "valence_recall": [0.5, 0.5],
    }


def test_round_trip_preserves_metrics_and_state(tmp_train_state_dir: Path) -> None:
    config = TrainingConfig(model_type="eegnet", epochs=10, seed=42)
    key = _run_key(config, n_splits=5)
    paths = _resume_paths(config.model_type, key)

    fold_metrics = [_sample_metrics(0.45), _sample_metrics(0.52), _sample_metrics(0.48)]
    best_state = {"layer.weight": torch.tensor([1.0, 2.0, 3.0])}

    _write_resume_state(
        paths,
        run_key=key,
        fold_metrics=fold_metrics,
        completed_folds=[0, 1, 2],
        best_fold=1,
        best_f1=0.52,
        best_state=best_state,
    )

    loaded = _load_resume_state(paths, expected_run_key=key)
    assert loaded is not None
    loaded_metrics, loaded_best_fold, loaded_state, loaded_done = loaded

    assert loaded_done == {0, 1, 2}
    assert loaded_best_fold == 1
    assert len(loaded_metrics) == 3
    assert loaded_metrics[1]["avg_macro_f1"] == pytest.approx(0.52)
    assert loaded_state is not None
    assert torch.equal(loaded_state["layer.weight"], best_state["layer.weight"])


def test_stale_run_key_returns_none(tmp_train_state_dir: Path) -> None:
    config = TrainingConfig(model_type="eegnet", epochs=10, seed=42)
    key = _run_key(config, n_splits=5)
    paths = _resume_paths(config.model_type, key)

    _write_resume_state(
        paths,
        run_key=key,
        fold_metrics=[_sample_metrics(0.40)],
        completed_folds=[0],
        best_fold=0,
        best_f1=0.40,
        best_state=None,
    )

    # A different n_splits flips the run_key — same file, but the caller is
    # asking about a different training contract, so resume must refuse.
    other_key = _run_key(config, n_splits=10)
    assert _load_resume_state(paths, expected_run_key=other_key) is None


def test_missing_file_returns_none(tmp_train_state_dir: Path) -> None:
    paths = _resume_paths("eegnet", "deadbeef0000")
    assert _load_resume_state(paths, expected_run_key="deadbeef0000") is None


def test_corrupt_json_returns_none(tmp_train_state_dir: Path) -> None:
    """A half-written / garbage JSON file must not crash resume."""
    paths = _resume_paths("eegnet", "cafebabe0000")
    paths.json_path.parent.mkdir(parents=True, exist_ok=True)
    paths.json_path.write_bytes(b"{not valid json")
    assert _load_resume_state(paths, expected_run_key="cafebabe0000") is None


def test_missing_pt_invalidates_resume(tmp_train_state_dir: Path) -> None:
    """JSON-without-.pt must refuse to resume.

    The partial-resume path would let a run finish with best_model_state=None
    and silently skip the final checkpoint save, so the loader rejects it
    outright and forces a clean restart.
    """
    config = TrainingConfig(model_type="eegnet", epochs=10, seed=42)
    key = _run_key(config, n_splits=5)
    paths = _resume_paths(config.model_type, key)

    _write_resume_state(
        paths,
        run_key=key,
        fold_metrics=[_sample_metrics(0.44)],
        completed_folds=[0],
        best_fold=0,
        best_f1=0.44,
        best_state={"w": torch.tensor([1.0])},
    )
    # Simulate the .pt going missing (partial volume sync, manual deletion).
    paths.state_path.unlink()

    assert _load_resume_state(paths, expected_run_key=key) is None


def test_clear_is_idempotent(tmp_train_state_dir: Path) -> None:
    paths = _resume_paths("eegnet", "feedface0000")
    # Clearing a non-existent state must not raise.
    _clear_resume_state(paths)

    _write_resume_state(
        paths,
        run_key="feedface0000",
        fold_metrics=[_sample_metrics(0.33)],
        completed_folds=[0],
        best_fold=0,
        best_f1=0.33,
        best_state={"w": torch.zeros(1)},
    )
    assert paths.json_path.exists()
    assert paths.state_path.exists()

    _clear_resume_state(paths)
    assert not paths.json_path.exists()
    assert not paths.state_path.exists()


def test_run_key_stable_across_irrelevant_hyperparams() -> None:
    """Tuning patience / LR must not invalidate an in-progress LOSO run."""
    a = TrainingConfig(model_type="cbramod", epochs=50, patience=20, lr=1e-3, seed=42)
    b = TrainingConfig(model_type="cbramod", epochs=50, patience=10, lr=5e-4, seed=42)
    assert _run_key(a, n_splits=32) == _run_key(b, n_splits=32)


def test_run_key_changes_on_model_type() -> None:
    a = TrainingConfig(model_type="eegnet", epochs=50, seed=42)
    b = TrainingConfig(model_type="cbramod", epochs=50, seed=42)
    assert _run_key(a, n_splits=32) != _run_key(b, n_splits=32)
