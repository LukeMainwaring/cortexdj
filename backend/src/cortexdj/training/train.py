"""The training loop and ``train_participant``: one participant's fold models, trained on their own trials only."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from cortexdj.ml.band_features import differential_entropy
from cortexdj.ml.checkpoint import AxisScores, ParticipantModels
from cortexdj.ml.dataset import load_deap_participant, load_participant_ratings
from cortexdj.ml.quadrant_classifier import AXES, QuadrantClassifier
from cortexdj.ml.recipe import RECIPE, TrainingRecipe
from cortexdj.training.folds import balanced_accuracy, stratified_folds

type Windows = npt.NDArray[np.floating[Any]]


def train_participant(participant_id: int, data_dir: Path, recipe: TrainingRecipe = RECIPE) -> ParticipantModels:
    """Train ``participant_id``'s fold models from ``sNN.dat`` and ``participant_ratings.csv`` in ``data_dir``.

    Labels come from the published ratings, cut at the recipe's threshold. One fold
    assignment, stratified on the quadrant of each trial's ratings, serves both axes. A plain
    function of its arguments: it reads no environment or global state and runs on CPU, so a
    batch runner or a recipe search can call it as it is.
    """
    eeg, _ = load_deap_participant(data_dir / f"s{participant_id:02d}.dat")
    ratings = load_participant_ratings(participant_id, data_dir)
    high = np.column_stack([ratings.arousal, ratings.valence]) >= recipe.label_threshold  # AXES order
    fold_of_trial = stratified_folds(high[:, 0] * 2 + high[:, 1], recipe.n_folds, recipe.seed)

    models, metrics = cross_validate(
        windows_of(eeg, recipe),
        high,
        fold_of_trial,
        make_model=QuadrantClassifier.fitted_to,
        to_input=lambda windows: differential_entropy(windows, recipe),
        recipe=recipe,
    )
    return ParticipantModels(models=models, fold_of_trial=fold_of_trial.tolist(), metrics=metrics)


def windows_of(eeg: Windows, recipe: TrainingRecipe = RECIPE) -> Windows:
    """Cut ``(trials, channels, samples)`` into non-overlapping windows: ``(trials, windows, channels, window_samples)``."""
    n_trials, n_channels, n_samples = eeg.shape
    n_windows = n_samples // recipe.window_samples
    cut = eeg[..., : n_windows * recipe.window_samples].reshape(n_trials, n_channels, n_windows, -1)
    return cut.transpose(0, 2, 1, 3)


def cross_validate[ModelT: nn.Module](
    windows: Windows,
    high: npt.NDArray[np.bool_],
    fold_of_trial: npt.NDArray[np.int64],
    *,
    make_model: Callable[[torch.Tensor], ModelT],
    to_input: Callable[[Windows], npt.NDArray[np.floating[Any]]],
    recipe: TrainingRecipe = RECIPE,
) -> tuple[list[ModelT], AxisScores]:
    """Fit one model per fold and score every window by the model that held its trial out.

    Takes any torch module with one logit per axis: ``make_model`` builds it from the training
    fold's inputs, and ``to_input`` maps windows (any leading axes) to that module's input.
    ``high`` is ``(trials, 2)`` in ``AXES`` order. Balanced accuracy is scored per window over the
    pooled held-out windows, not per fold, so a held-out fold may lack a class on an axis.
    """
    inputs = torch.as_tensor(to_input(windows), dtype=torch.float32)  # (trials, windows, features)
    n_windows = inputs.shape[1]
    window_high = np.repeat(high, n_windows, axis=0)
    held_out_logits = np.empty((len(high), n_windows, len(AXES)), dtype=np.float32)

    models = []
    for fold in range(recipe.n_folds):
        train, test = fold_of_trial != fold, fold_of_trial == fold
        for axis, name in enumerate(AXES):
            if len(np.unique(high[train, axis])) < 2:
                raise ValueError(
                    f"Fold {fold}'s training trials are all {'high' if high[train, axis].all() else 'low'} "
                    f"on {name}, so it cannot learn that axis"
                )
        model = _fit(make_model, inputs[train].flatten(0, 1), torch.as_tensor(high[train]), n_windows, recipe)
        with torch.no_grad():
            held_out_logits[test] = model(inputs[test].flatten(0, 1)).reshape(-1, n_windows, len(AXES)).numpy()
        models.append(model)

    predicted = held_out_logits.reshape(-1, len(AXES)) > 0
    arousal, valence = (balanced_accuracy(window_high[:, axis], predicted[:, axis]) for axis in range(len(AXES)))
    return models, AxisScores(arousal=arousal, valence=valence, mean=(arousal + valence) / 2)


def _fit[ModelT: nn.Module](
    make_model: Callable[[torch.Tensor], ModelT],
    inputs: torch.Tensor,
    trial_high: torch.Tensor,
    n_windows: int,
    recipe: TrainingRecipe,
) -> ModelT:
    """Full-batch AdamW on class-balanced binary cross-entropy for a fixed number of epochs, no early stopping.

    Runs inside a forked RNG seeded from the recipe, so a rerun gives the same model without
    touching torch's global random state.
    """
    targets = trial_high.repeat_interleave(n_windows, dim=0).float()
    positives = targets.sum(dim=0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(len(targets) - positives) / positives)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(recipe.seed)
        model = make_model(inputs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay)
        model.train()
        for _ in range(recipe.epochs):
            optimizer.zero_grad()
            loss_fn(model(inputs), targets).backward()
            optimizer.step()
    model.eval()
    return model
