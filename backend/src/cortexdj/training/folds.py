"""Folds and the metric: a participant's trials split for held-out scoring, and balanced accuracy."""

from typing import Any

import numpy as np
import numpy.typing as npt


def stratified_folds(strata: npt.NDArray[np.integer[Any]], n_folds: int, seed: int) -> npt.NDArray[np.int64]:
    """Assign each trial a fold, spreading every stratum's trials as evenly as the folds allow.

    ``strata`` is one class per trial (the quadrant of its ratings), and the result is that
    trial's fold. Folds are per trial, never per window, so a trial's windows never straddle a
    split, and each trial is held out by exactly one fold. Dealing every stratum round-robin
    from where the last one stopped keeps the folds' sizes within one of each other too.
    """
    rng = np.random.default_rng(seed)
    fold_of_trial = np.empty(len(strata), dtype=np.int64)
    dealt = 0
    for stratum in np.unique(strata):
        members = rng.permutation(np.flatnonzero(strata == stratum))
        fold_of_trial[members] = (dealt + np.arange(len(members))) % n_folds
        dealt += len(members)
    return fold_of_trial


def balanced_accuracy(truth: npt.NDArray[np.bool_], predicted: npt.NDArray[np.bool_]) -> float:
    """The mean of the recall on each class present in ``truth``: 0.50 is chance whatever the imbalance."""
    recalls = [np.mean(predicted[truth == label] == label) for label in (False, True) if np.any(truth == label)]
    return float(np.mean(recalls))
