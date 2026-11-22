"""The cross-subject baseline script: logistic regression on differential entropy, across and within participants.

Kept as evidence, not as part of the training path. It re-runs the measurement behind the
quadrant classifier's design: a model trained on other participants scores at chance on a
held-out one without calibration (balanced accuracy 0.50 on both axes), while a participant's
own trials carry some signal (about 0.60 for valence, 0.55 for arousal). That is why
``train-classifier`` trains each participant on their own trials and has no
leave-one-subject-out mode; do not add one there, re-run this script instead.

It prints one markdown table: for valence and arousal, under the fixed ``>= 5`` rule and each
participant's own median, the balanced accuracy, plain accuracy and majority-class accuracy of
leave-one-subject-out (with train-set scaling, and with each participant's features z-scored over
their own windows, which uses no labels) and of a stratified 5-fold within-subject split by
trial (a trial's windows stay on one side). Scores are per window, averaged over participants.
It takes about 40 seconds on a laptop CPU for all 32 participants.

Labels come from ``participant_ratings.csv`` through ``load_participant_ratings``, never from the
``.dat`` files' label array. Its features filter each whole 60-second trial and then cut it
into 4-second windows, unlike ``band_features``, which filters each window: that is how the
recorded numbers were made, so keep it, and do not reuse this as a feature path.

This is the only file that imports scikit-learn. It is not declared as a dependency because
braindecode already installs it; nothing imports this module and it has no test.

Usage:
    uv run --directory backend python -m cortexdj.training.cross_subject_baseline data/deap
"""

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from cortexdj.ml.dataset import load_deap_participant, load_participant_ratings

type Floats = npt.NDArray[np.floating[Any]]
type Ints = npt.NDArray[np.int64]
type Scores = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]

N_PARTICIPANTS = 32
N_TRIALS = 40
SAMPLE_RATE = 128
WINDOW_SAMPLES = 4 * SAMPLE_RATE  # 15 windows a trial, no overlap
# No delta: DEAP's preprocessed files are already band-passed at 4 to 45 Hz.
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
LABEL_RULES = {"`>= 5`": "threshold", "`>` own median": "median"}


def de_features(eeg: Floats) -> Floats:
    """``(trials, 32, 7680)`` EEG to ``(trials, 15, 128)`` differential entropy per window, channel-major."""
    per_band = []
    for low, high in BANDS.values():
        sos = signal.butter(4, [low, high], btype="band", fs=SAMPLE_RATE, output="sos")
        filtered = signal.sosfiltfilt(sos, eeg, axis=-1)
        windows = filtered.reshape(*eeg.shape[:2], -1, WINDOW_SAMPLES)
        # DE of a Gaussian band signal: half the log of 2 pi e times the variance.
        per_band.append(0.5 * np.log(2 * np.pi * np.e * windows.var(axis=-1)))
    stacked = np.stack(per_band, axis=-1)  # (trials, channels, windows, bands)
    features: Floats = stacked.transpose(0, 2, 1, 3).reshape(eeg.shape[0], stacked.shape[2], -1)
    return features


def binarize(ratings: Floats, rule: str) -> Ints:
    """``(participants, trials)`` ratings on one axis to 0/1 labels: at or above 5, or above the participant's median."""
    if rule == "threshold":
        return (ratings >= 5).astype(np.int64)
    return (ratings > np.median(ratings, axis=1, keepdims=True)).astype(np.int64)


def fit_predict(x_train: Floats, y_train: Ints, x_test: Floats) -> Ints:
    """Standardize on the training set, fit a class-balanced logistic regression, predict the test set."""
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(x_train), y_train)
    return np.asarray(model.predict(scaler.transform(x_test)), dtype=np.int64)


def majority_hits(y_train: Ints, y_test: Ints) -> npt.NDArray[np.bool_]:
    """Which test labels the training set's majority class gets right."""
    return np.asarray(y_test == np.bincount(y_train, minlength=2).argmax())


def leave_one_subject_out(features: Floats, labels: Ints, per_participant_norm: bool) -> Scores:
    """Balanced accuracy, accuracy and majority accuracy for each held-out participant.

    ``features`` is ``(participants, trials, windows, 128)``, ``labels`` ``(participants, trials)``.
    """
    if per_participant_norm:
        mean = features.mean(axis=(1, 2), keepdims=True)
        std = features.std(axis=(1, 2), keepdims=True)
        features = (features - mean) / std
    n_participants, n_trials, n_windows, _ = features.shape
    x = features.reshape(n_participants, n_trials * n_windows, -1)
    y = np.repeat(labels, n_windows, axis=1)
    bal, acc, maj = [], [], []
    for held_out in range(n_participants):
        train = np.arange(n_participants) != held_out
        x_train, y_train = x[train].reshape(-1, x.shape[-1]), y[train].reshape(-1)
        pred = fit_predict(x_train, y_train, x[held_out])
        # A participant with one class present scores recall on that class only.
        bal.append(float(balanced_accuracy_score(y[held_out], pred)))
        acc.append(float((pred == y[held_out]).mean()))
        maj.append(float(majority_hits(y_train, y[held_out]).mean()))
    return np.array(bal), np.array(acc), np.array(maj)


def within_subject(features: Floats, labels: Ints) -> tuple[Scores, int]:
    """Stratified 5-fold over each participant's trials, and how many participants had too few trials in a class."""
    n_participants, n_trials, n_windows, _ = features.shape
    bal, acc, maj, skipped = [], [], [], 0
    window_trial = np.repeat(np.arange(n_trials), n_windows)
    for p in range(n_participants):
        y_trials = labels[p]
        n_splits = min(5, int(np.bincount(y_trials, minlength=2).min()))
        if n_splits < 2:
            skipped += 1
            continue
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        y = np.repeat(y_trials, n_windows)
        x = features[p].reshape(n_trials * n_windows, -1)
        pred = np.empty(n_trials * n_windows, dtype=np.int64)
        hits = np.empty(n_trials * n_windows, dtype=bool)
        for train_trials, test_trials in folds.split(np.zeros(n_trials), y_trials):
            train = np.isin(window_trial, train_trials)
            test = np.isin(window_trial, test_trials)
            pred[test] = fit_predict(x[train], y[train], x[test])
            hits[test] = majority_hits(y[train], y[test])
        bal.append(float(balanced_accuracy_score(y, pred)))
        acc.append(float((pred == y).mean()))
        maj.append(float(hits.mean()))
    return (np.array(bal), np.array(acc), np.array(maj)), skipped


def load(data_dir: Path) -> tuple[Floats, dict[str, Floats]]:
    """Every participant's window features and their published ratings by axis, both indexed by participant and trial."""
    features, valence, arousal = [], [], []
    for participant_id in range(1, N_PARTICIPANTS + 1):
        eeg, eeg_labels = load_deap_participant(data_dir / f"s{participant_id:02d}.dat")
        ratings = load_participant_ratings(participant_id, data_dir, eeg_labels=eeg_labels)
        features.append(de_features(eeg))
        valence.append(ratings.valence)
        arousal.append(ratings.arousal)
    return np.stack(features), {"valence": np.stack(valence), "arousal": np.stack(arousal)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score logistic regression on DEAP band features across and within participants"
    )
    parser.add_argument("data_dir", type=Path, help="directory holding s01.dat to s32.dat and participant_ratings.csv")
    args = parser.parse_args()

    started = time.perf_counter()
    features, ratings = load(args.data_dir)
    feature_seconds = time.perf_counter() - started
    print(f"features {features.shape} in {feature_seconds:.1f}s\n")  # noqa: T201

    print("| split | axis | label rule | balanced acc (mean ± sd) | accuracy | majority acc |")  # noqa: T201
    print("|---|---|---|---|---|---|")  # noqa: T201
    fit_started = time.perf_counter()
    for axis, axis_ratings in ratings.items():
        for rule_name, rule in LABEL_RULES.items():
            labels = binarize(axis_ratings, rule)
            within, skipped = within_subject(features, labels)
            splits: dict[str, Scores] = {
                "leave-one-subject-out": leave_one_subject_out(features, labels, False),
                "leave-one-subject-out, per-participant z-score": leave_one_subject_out(features, labels, True),
                "within-subject by trial": within,
            }
            for split, (bal, acc, maj) in splits.items():
                print(  # noqa: T201
                    f"| {split} | {axis} | {rule_name} | {bal.mean():.3f} ± {bal.std():.3f} "
                    f"| {acc.mean():.3f} | {maj.mean():.3f} |"
                )
            if skipped:
                print(  # noqa: T201
                    f"<!-- within-subject {axis} {rule_name}: skipped {skipped} participants "
                    "with under 2 trials in a class -->"
                )
    fit_seconds = time.perf_counter() - fit_started
    total_seconds = time.perf_counter() - started
    print(f"\nfeatures {feature_seconds:.1f}s, fits {fit_seconds:.1f}s, total {total_seconds:.1f}s")  # noqa: T201


if __name__ == "__main__":
    main()
