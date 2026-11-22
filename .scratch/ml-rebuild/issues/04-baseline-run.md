# 04: A baseline run on band features

**Type:** task
**Status:** resolved

## Question

Run a throwaway baseline so later decisions rest on real numbers: logistic regression on differential-entropy band features from the 32 DEAP files in `backend/data/deap`, for arousal and valence, under leave-one-subject-out and under a within-subject split by trial. Report balanced accuracy and the majority baseline for each, with the fixed threshold at 5 and with each participant's median as the label rule, and how long it took. The code lives on a `prototype/` branch and is not the build.

## Answer

Resolved 2026-11-22. Code: `prototypes/ml-rebuild-baseline/baseline.py` on branch `refs/heads/prototype/ml-rebuild-baseline` (commit bcfe9dd, local only, never merged). Run with `uv run --directory backend python <path>/baseline.py data/deap`.

**Setup.** Labels from `participant_ratings.csv` (ticket 09). 32 EEG channels, the 60 seconds after the baseline, cut into 4-second windows with no overlap (15 a trial, 19,200 in all). Features: differential entropy in theta, alpha, beta and gamma for each channel (128 per window; no delta, the files are band-passed at 4 to 45 Hz). Model: scikit-learn logistic regression, `C=1.0`, balanced class weights, features standardized on the training set. Scores are per window, averaged over the 32 participants. Within-subject is a stratified 5-fold over a participant's 40 trials, with a trial's windows kept on one side.

| split | axis | label rule | balanced acc (mean ± sd) | accuracy | majority acc |
|---|---|---|---|---|---|
| leave-one-subject-out | valence | `>= 5` | 0.499 ± 0.024 | 0.506 | 0.566 |
| leave-one-subject-out, per-participant z-score | valence | `>= 5` | 0.534 ± 0.079 | 0.534 | 0.566 |
| within-subject by trial | valence | `>= 5` | 0.596 ± 0.072 | 0.608 | 0.580 |
| leave-one-subject-out | valence | `>` own median | 0.499 ± 0.026 | 0.498 | 0.505 |
| leave-one-subject-out, per-participant z-score | valence | `>` own median | 0.538 ± 0.086 | 0.538 | 0.505 |
| within-subject by trial | valence | `>` own median | 0.606 ± 0.075 | 0.606 | 0.505 |
| leave-one-subject-out | arousal | `>= 5` | 0.507 ± 0.018 | 0.469 | 0.589 |
| leave-one-subject-out, per-participant z-score | arousal | `>= 5` | 0.524 ± 0.053 | 0.526 | 0.589 |
| within-subject by trial | arousal | `>= 5` | 0.551 ± 0.058 | 0.583 | 0.642 |
| leave-one-subject-out | arousal | `>` own median | 0.499 ± 0.027 | 0.499 | 0.512 |
| leave-one-subject-out, per-participant z-score | arousal | `>` own median | 0.523 ± 0.046 | 0.522 | 0.512 |
| within-subject by trial | arousal | `>` own median | 0.546 ± 0.063 | 0.546 | 0.512 |

- **Cross-subject is chance** with plain scaling (0.50 on both axes, both rules). Z-scoring each participant's features over their own windows, which uses no labels, lifts it to 0.52 to 0.54. That extra row was not in the question; it was cheap and bears on the preprocessing ticket.
- **Within-subject carries real signal for valence** (0.60) and less for arousal (0.55). With trial labels shuffled inside each participant, the same split scores 0.500 and 0.499 over 5 repeats, so the split does not leak.
- **The label rule barely moves balanced accuracy** (at most 0.01). It moves the majority baseline a lot: plain accuracy beats it only under the median rule, and under `>= 5` only for within-subject valence.
- **Time:** 40 seconds on this laptop's CPU for everything (16 for features, 24 for 288 leave-one-subject-out fits and 640 within-subject fits). Memory is small.
- **Limits:** one seed for the within-subject folds; no tuning of `C`, bands or window length; window-level scores only, no vote over a trial; the EEG is unverified against an official copy (ticket 09).

## Comments

- 2026-11-22: ticket 09 resolved. Take labels from `backend/data/deap/participant_ratings.csv` joined on `Participant_id` and `Experiment_id` (trial index + 1), never from the `.dat` files' `labels` array, whose valence and arousal are altered in 439 trials.
