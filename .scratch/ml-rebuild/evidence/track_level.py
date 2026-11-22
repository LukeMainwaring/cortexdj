"""Throwaway for ml-rebuild ticket 12: session numbers when the unit is the track."""
import sys
from pathlib import Path
import numpy as np
from score_spread import fit_sklearn, held_out_scores, load

features, labels = load(Path(sys.argv[1]))
for c in (1.0, 0.01):
    val = held_out_scores(features, labels[:, :, 0], fit_sklearn(c)).reshape(32, 40, 15).mean(axis=2)
    aro = held_out_scores(features, labels[:, :, 1], fit_sklearn(c)).reshape(32, 40, 15).mean(axis=2)
    quad = (aro >= 0.5) * 2 + (val >= 0.5)
    true = labels[:, :, 1] * 2 + labels[:, :, 0]
    trans = (quad[:, 1:] != quad[:, :-1]).sum(axis=1)
    ttrans = (true[:, 1:] != true[:, :-1]).sum(axis=1)
    dom = np.array([np.bincount(q, minlength=4).max() / 40 for q in quad])
    tdom = np.array([np.bincount(q, minlength=4).max() / 40 for q in true])
    lab = ["throughout" if d >= 0.9 else "mostly" if d >= 0.6 else "mixed" for d in dom]
    tlab = ["throughout" if d >= 0.9 else "mostly" if d >= 0.6 else "mixed" for d in tdom]
    domq = np.array([np.bincount(q, minlength=4).argmax() for q in quad])
    tdomq = np.array([np.bincount(q, minlength=4).argmax() for q in true])
    print(f"C={c}: track quadrant right {np.mean(quad == true):.3f}")
    print(f"  transitions of 39: model mean {trans.mean():.0f} ({trans.min()}-{trans.max()}); ratings {ttrans.mean():.0f} ({ttrans.min()}-{ttrans.max()})")
    print(f"  dominant share: model {dom.mean():.2f} ({dom.min():.2f}-{dom.max():.2f}); ratings {tdom.mean():.2f} ({tdom.min():.2f}-{tdom.max():.2f})")
    print(f"  card labels model { {k: lab.count(k) for k in set(lab)} } ratings { {k: tlab.count(k) for k in set(tlab)} }")
    print(f"  dominant quadrant counts (calm, relaxed, stressed, excited): model {np.bincount(domq, minlength=4)} ratings {np.bincount(tdomq, minlength=4)}; match {np.mean(domq == tdomq):.2f}")
    print(f"  track score quantiles 5/25/50/75/95: val {np.round(np.quantile(val, [.05,.25,.5,.75,.95]),2)} aro {np.round(np.quantile(aro, [.05,.25,.5,.75,.95]),2)}")
