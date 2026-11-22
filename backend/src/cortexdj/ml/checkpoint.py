"""The quadrant-classifier checkpoint: what it holds, and the one place its file layout is read and written.

One file, made locally by ``train-classifier`` and never downloaded. For each participant it
holds the five fold models and which trial each fold held out, plus balanced accuracy per
participant and across participants, and the training recipe. It stores no predictions: a
score is always recomputed from the EEG by the fold model that held that trial out.
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cortexdj.core.paths import CHECKPOINTS_DIR
from cortexdj.ml.quadrant_classifier import QuadrantClassifier
from cortexdj.ml.recipe import RECIPE, TrainingRecipe

CHECKPOINT_PATH = CHECKPOINTS_DIR / "quadrant_classifier.pt"


@dataclass(frozen=True)
class AxisScores:
    """Balanced accuracy on each axis and their mean, the single number. Chance is 0.50 for all three."""

    arousal: float
    valence: float
    mean: float


@dataclass(frozen=True)
class Spread:
    """A figure averaged over participants, with its standard deviation across them."""

    mean: float
    std: float


@dataclass(frozen=True)
class ScoreSummary:
    """The three figures ``train-classifier`` prints, each with its spread across participants."""

    arousal: Spread
    valence: Spread
    mean: Spread

    @classmethod
    def of(cls, scores: list[AxisScores]) -> ScoreSummary:
        def spread(values: list[float]) -> Spread:
            return Spread(mean=float(np.mean(values)), std=float(np.std(values)))

        return cls(
            arousal=spread([s.arousal for s in scores]),
            valence=spread([s.valence for s in scores]),
            mean=spread([s.mean for s in scores]),
        )


@dataclass(frozen=True)
class ParticipantModels:
    """One participant's fold models: ``models[k]`` was trained on every trial whose ``fold_of_trial`` is not ``k``.

    ``fold_of_trial[e]`` is the fold that held out trial ``e`` (EEG file order), so the model
    that never saw trial ``e`` is ``models[fold_of_trial[e]]``. ``metrics`` is scored over the
    participant's pooled held-out windows.
    """

    models: list[QuadrantClassifier]
    fold_of_trial: list[int]
    metrics: AxisScores


@dataclass(frozen=True)
class QuadrantCheckpoint:
    """A loaded checkpoint: every trained participant, the summary across them, and the recipe they share."""

    recipe: TrainingRecipe
    participants: dict[int, ParticipantModels]
    summary: ScoreSummary

    @classmethod
    def of(cls, participants: dict[int, ParticipantModels], recipe: TrainingRecipe = RECIPE) -> QuadrantCheckpoint:
        return cls(
            recipe=recipe,
            participants=participants,
            summary=ScoreSummary.of([p.metrics for p in participants.values()]),
        )


def save_checkpoint(checkpoint: QuadrantCheckpoint, path: Path = CHECKPOINT_PATH) -> None:
    """Write ``checkpoint`` to ``path``, replacing any file there: a subset run holds only its participants.

    State dicts and plain values only, so ``load_checkpoint`` can use weights-only loading.
    """
    payload = {
        "recipe": asdict(checkpoint.recipe),
        "summary": asdict(checkpoint.summary),
        "participants": {
            participant_id: {
                "models": [model.state_dict() for model in trained.models],
                "fold_of_trial": list(trained.fold_of_trial),
                "metrics": asdict(trained.metrics),
            }
            for participant_id, trained in checkpoint.participants.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path = CHECKPOINT_PATH, recipe: TrainingRecipe = RECIPE) -> QuadrantCheckpoint:
    """Read the checkpoint at ``path``, refusing one made with a recipe other than ``recipe``.

    The stored recipe is compared with the code's value by value, and the error names the first
    setting that differs: a model read under settings it was not trained with scores garbage
    without failing, so do not relax this to a warning.
    """
    if not path.exists():
        raise FileNotFoundError(f"No quadrant-classifier checkpoint at {path}; run train-classifier to make one")
    payload = torch.load(path, weights_only=True)
    _check_recipe(payload.get("recipe", {}), recipe, path)

    participants = {}
    for participant_id, stored in payload["participants"].items():
        models = []
        for state in stored["models"]:
            model = QuadrantClassifier(n_features=state["standardize.mean"].shape[0])
            model.load_state_dict(state)
            model.eval()
            models.append(model)
        participants[int(participant_id)] = ParticipantModels(
            models=models,
            fold_of_trial=[int(fold) for fold in stored["fold_of_trial"]],
            metrics=AxisScores(**stored["metrics"]),
        )
    summary = payload["summary"]
    return QuadrantCheckpoint(
        recipe=recipe,
        participants=participants,
        summary=ScoreSummary(**{axis: Spread(**summary[axis]) for axis in ("arousal", "valence", "mean")}),
    )


def _check_recipe(stored: dict[str, Any], recipe: TrainingRecipe, path: Path) -> None:
    current = asdict(recipe)
    for field in fields(recipe):
        stored_value = stored.get(field.name, "<missing>")
        if stored_value != current[field.name]:
            raise ValueError(
                f"The checkpoint at {path} was trained with {field.name}={stored_value!r}, but the code's "
                f"training recipe has {field.name}={current[field.name]!r}; run train-classifier again"
            )
