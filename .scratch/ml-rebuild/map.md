# Map: Rebuild the signal pipeline and the quadrant classifier

Label: wayfinder:map

## Destination

Settled decisions for one spec: the DEAP signal pipeline and the quadrant classifier rebuilt together, with training code back in the repo. One classifier pipeline the user can explain end to end, reported with an honest number (balanced accuracy against the majority baseline, on a stated split), even if that number stays near chance. The map is done when nothing is left to decide before `to-spec`.

## Notes

- Domain: EEG emotion decoding on DEAP. Read `GLOSSARY.md` (Segment, Quadrant, Session) and `backend/src/cortexdj/ml/` before a ticket. Grilling tickets use the `grilling` and `domain-modeling` skills; research tickets use `research`.
- Simplicity outranks accuracy. Prefer the approach that is easier to read and explain over one that gains a point or two.
- One code path. Training is one plain Python function that runs a single fold on any machine in minutes; Modal is a thin wrapper that fans folds out for full runs. No multiplying of device, config, or backend branches: few knobs, one model, one preprocessing path.
- Exactly one classifier pipeline survives; the other is deleted along with the `isinstance` branch in `ml/predict.py`.
- The app contract is fixed by default: 4-second stored segments, arousal and valence scores from 0 to 1, a quadrant, band powers; no migrations, no frontend changes. Reopen it only if a ticket finds the segment timing or the scores are plainly wrong. The model's input window may differ from the stored segment.
- Input stays DEAP's preprocessed `.dat` files (128Hz, already filtered, eye artifacts removed).
- The contrastive encoder is untouched: `contrastive_best.pt` must keep working, so `trial_to_eeg_windows` and `CBRAMOD_SCALE_FACTOR` as retrieval uses them do not change.
- Starting facts: the libraries are current (torch 2.15, braindecode 2.0, mne 1.13, scipy 1.18); the old training code exists only in git history (`legacy/backend/src/cortexdj/ml/train.py`, `legacy/backend/scripts/modal_train.py`, `legacy/backend/autoresearch/`, removed in 2378eff); the CBraMod checkpoint records about 52% leave-one-subject-out.
- `after-port 01` (every seeded segment is classified calm) is folded into this effort: the new checkpoint fixes it, it is not patched on its own. Other open tickets may go stale the same way; say so when a decision makes one obsolete.

## Decisions so far

- [Modal's current setup for small fan-out training runs](issues/03-modal-setup-for-small-fan-out-runs.md): a 30-line wrapper around a plain `train_fold`, CPU by default, folds via `.map`, data on a read-only Volume; about $0.67 per 32-fold run at an assumed 5 minutes a fold; untested on Modal.
- [Current braindecode and MNE practice for DEAP](issues/01-current-braindecode-mne-practice-for-deap.md): no loader to adopt, so we keep wrapping the arrays; DEAP already filtered, re-referenced and baseline-corrected, leaving normalization, windowing and labels to us; the delta band is empty; small braindecode models are single-output; the local labels look altered.
- [What recent benchmarks report for DEAP arousal and valence](issues/02-what-deap-benchmarks-report.md): trial-safe results sit near chance for every model family, foundation models included (about 0.50 to 0.57 balanced accuracy); numbers above 90% come from leaking window-level splits; and the local labels differ from the published ratings in 439 trials.
- [Are the local DEAP labels the official ones?](issues/09-are-the-local-deap-labels-official.md): no, the Kaggle mirror re-saved the files with valence and arousal changed in 439 trials; the effort keeps the local EEG (unverified against an official copy, accepted) and takes labels from `participant_ratings.csv` in `backend/data/deap/`, joined on participant and `Experiment_id`; the official site is down.
- [A baseline run on band features](issues/04-baseline-run.md): logistic regression on differential entropy scores chance cross-subject (0.50 balanced accuracy, about 0.53 with per-participant z-scoring) and 0.60 valence, 0.55 arousal within-subject by trial; the label rule barely moves balanced accuracy; the whole run takes 40 seconds on a laptop.

## Not yet specified

- Autoresearch: once a training script and one metric exist, how a `/goal`-style loop runs against them, and whether it is worth having at all. Last leg, kept light.
- Seeding after the new checkpoint: re-seeding the sessions, what a session should look like in the app when the honest number is near chance, and whether the stored scores need calibrating.
- Whether the app contract has to reopen (segment length, score meaning), if the label or preprocessing tickets find it plainly wrong.
- Band powers shown in the app: whether they come from the same preprocessing path as the model's input or stay a separate Welch computation.
- How training code is tested and checked in CI without a GPU or the DEAP files.
- What goes stale: `after-port 01`, `docs/ROADMAP.md`, the glossary's Segment and Quadrant entries, the architecture guide pages, the checkpoint download script, `backend/data/DEAP_SETUP.md` (its "same preprocessed data" and "~24/76" claims, and the missing ratings-file step), the recorded 52% figure.

## Out of scope

- Retraining the contrastive EEG↔CLAP encoder: blocked on `audio-decoding 01` and a long run; a fresh effort later.
- Going back to DEAP's raw recordings.
- Other datasets than DEAP.
