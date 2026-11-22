"""Throwaway for ml-rebuild ticket 12: what held-out scores look like in a session.

Same features, labels and folds as the ticket 04 baseline; keeps the held-out
probabilities instead of only the hard predictions.
Usage: python score_spread.py <deap dir>
"""

import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

FS = 128
BANDS = [(4, 8), (8, 13), (13, 30), (30, 45)]
QUADRANT = {(1, 1): "excited", (1, 0): "stressed", (0, 1): "relaxed", (0, 0): "calm"}  # (arousal, valence)


def load(data_dir: Path):
    ratings = np.zeros((32, 40, 2))
    with open(data_dir / "participant_ratings.csv") as f:
        for row in csv.DictReader(f):
            ratings[int(row["Participant_id"]) - 1, int(row["Experiment_id"]) - 1] = (
                float(row["Valence"]),
                float(row["Arousal"]),
            )
    feats = []
    for p in range(1, 33):
        with open(data_dir / f"s{p:02d}.dat", "rb") as f:
            eeg = pickle.load(f, encoding="latin1")["data"][:, :32, 3 * FS :]
        per_band = []
        for low, high in BANDS:
            sos = signal.butter(4, [low, high], btype="band", fs=FS, output="sos")
            w = signal.sosfiltfilt(sos, eeg, axis=-1).reshape(40, 32, -1, 4 * FS)
            per_band.append(0.5 * np.log(2 * np.pi * np.e * w.var(axis=-1)))
        s = np.stack(per_band, axis=-1)
        feats.append(s.transpose(0, 2, 1, 3).reshape(40, 15, -1))
    return np.stack(feats), (ratings >= 5).astype(int)  # (32,40,15,128), (32,40,2) valence, arousal


def fit_sklearn(c):
    def fit(xtr, ytr, xte):
        m = LogisticRegression(C=c, max_iter=5000, class_weight="balanced").fit(xtr, ytr)
        return m.predict_proba(xte)[:, 1]

    return fit


def fit_torch(weight_decay, epochs=300, lr=0.01):
    def fit(xtr, ytr, xte):
        torch.manual_seed(0)
        x, y = torch.tensor(xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
        layer = torch.nn.Linear(x.shape[1], 1)
        pos_weight = (1 - y).sum() / y.sum()
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(layer.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss_fn(layer(x).squeeze(1), y).backward()
            opt.step()
        with torch.no_grad():
            return torch.sigmoid(layer(torch.tensor(xte, dtype=torch.float32)).squeeze(1)).numpy()

    return fit


def held_out_scores(features, labels, fit):
    """(32, 600) held-out probability of high for one axis, windows in trial order."""
    out = np.zeros((32, 600))
    trial_of = np.repeat(np.arange(40), 15)
    for p in range(32):
        x, y = features[p].reshape(600, -1), np.repeat(labels[p], 15)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(40), labels[p]):
            a, b = np.isin(trial_of, tr), np.isin(trial_of, te)
            sc = StandardScaler().fit(x[a])
            out[p, b] = fit(sc.transform(x[a]), y[a], sc.transform(x[b]))
    return out


def report(name, features, labels, fit):
    print(f"\n===== {name} =====")
    val = held_out_scores(features, labels[:, :, 0], fit)
    aro = held_out_scores(features, labels[:, :, 1], fit)
    yv, ya = np.repeat(labels[:, :, 0], 15, axis=1), np.repeat(labels[:, :, 1], 15, axis=1)

    for axis, s, y in (("valence", val, yv), ("arousal", aro, ya)):
        bal = np.mean([balanced_accuracy_score(y[p], s[p] >= 0.5) for p in range(32)])
        q = np.quantile(s, [0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"{axis}: balanced acc {bal:.3f}; score quantiles 5/25/50/75/95% = {np.round(q, 2)}")
        print(
            f"  share within 0.4-0.6: {np.mean((s > 0.4) & (s < 0.6)):.2f}; "
            f"below 0.1 or above 0.9: {np.mean((s < 0.1) | (s > 0.9)):.2f}"
        )
        edges = np.linspace(0, 1, 6)
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (s >= lo) & (s <= hi)
            rows.append(f"{lo:.1f}-{hi:.1f}: n={m.mean():.2f} actual high={y[m].mean():.2f}")
        print("  calibration (score bin: share of windows, share truly high): " + " | ".join(rows))
        # trial level: mean of 15 windows
        st, yt = s.reshape(32, 40, 15).mean(axis=2), y.reshape(32, 40, 15)[:, :, 0]
        print(f"  trial-mean balanced acc: {np.mean([balanced_accuracy_score(yt[p], st[p] >= 0.5) for p in range(32)]):.3f}")

    # sessions as the app would store them
    pa, pv = (aro >= 0.5).astype(int), (val >= 0.5).astype(int)
    quad_pred = pa * 2 + pv
    quad_true = ya * 2 + yv
    print(f"segment quadrant right (4-way): {np.mean(quad_pred == quad_true):.3f}; both axes wrong: {np.mean((pa != ya) & (pv != yv)):.3f}")
    trans = (quad_pred[:, 1:] != quad_pred[:, :-1]).sum(axis=1)
    idx = np.arange(1, 600)
    within = (quad_pred[:, 1:] != quad_pred[:, :-1])[:, idx % 15 != 0].sum(axis=1)
    true_trans = (quad_true[:, 1:] != quad_true[:, :-1]).sum(axis=1)
    print(f"transitions per session (of 599 steps): mean {trans.mean():.0f}, range {trans.min()}-{trans.max()}; inside a track: {within.mean():.0f}; if ratings were the truth: {true_trans.mean():.0f}")
    dom = np.array([np.bincount(quad_pred[p], minlength=4).max() / 600 for p in range(32)])
    dom_q = np.array([np.bincount(quad_pred[p], minlength=4).argmax() for p in range(32)])
    dom_true = np.array([np.bincount(quad_true[p], minlength=4).argmax() for p in range(32)])
    labels_ = ["throughout" if d >= 0.9 else "mostly" if d >= 0.6 else "mixed" for d in dom]
    print(f"dominant quadrant share per session: mean {dom.mean():.2f}, range {dom.min():.2f}-{dom.max():.2f}; card labels: { {k: labels_.count(k) for k in set(labels_)} }")
    print(f"sessions whose dominant quadrant matches the ratings' dominant quadrant: {np.mean(dom_q == dom_true):.2f}; distinct dominant quadrants across 32: {np.bincount(dom_q, minlength=4)} (calm, relaxed, stressed, excited)")
    # track level (SessionTrack): mean scores over 15 segments
    ta, tv = aro.reshape(32, 40, 15).mean(axis=2) >= 0.5, val.reshape(32, 40, 15).mean(axis=2) >= 0.5
    tq_true = quad_true.reshape(32, 40, 15)[:, :, 0]
    print(f"track quadrant right (4-way, mean of 15 segments): {np.mean((ta * 2 + tv) == tq_true):.3f}")
    step = np.sqrt(np.diff(val, axis=1) ** 2 + np.diff(aro, axis=1) ** 2)
    print(f"path length per session: mean {step.sum(axis=1).mean():.0f}; mean step {step.mean():.2f}")


def main():
    features, labels = load(Path(sys.argv[1]))
    report("sklearn logistic regression, C=1 (the ticket 04 baseline)", features, labels, fit_sklearn(1.0))
    report("sklearn logistic regression, C=0.01 (heavier regularization)", features, labels, fit_sklearn(0.01))
    report("torch nn.Linear, AdamW lr 0.01, 300 epochs, weight decay 0.01", features, labels, fit_torch(0.01))
    report("torch nn.Linear, AdamW lr 0.01, 2000 epochs, weight decay 0.01", features, labels, fit_torch(0.01, epochs=2000))


if __name__ == "__main__":
    main()
