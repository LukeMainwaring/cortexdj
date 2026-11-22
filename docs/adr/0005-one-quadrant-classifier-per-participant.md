# 0005: The quadrant classifier is calibrated to one participant, not shared across them

**Status:** accepted (2026-11-22)

## Context

EEG differs more between people than between emotions: head shape, cap placement, and how each
person uses the 1 to 9 rating scale. On DEAP, a model tested on a participant it never trained on
scores chance for arousal and valence. Our own baseline (logistic regression on differential
entropy) scored 0.50 balanced accuracy leave-one-subject-out, and published trial-safe results
say the same for every model family, frozen foundation models included (0.48 to 0.57). The same
baseline, trained on some of a participant's trials and tested on their others, scored 0.60 for
valence and 0.55 for arousal.

One model for everyone is the simpler thing to ship, and it is what the app did. But its honest
number is chance, and nothing built on a chance classifier is worth showing.

## Decision

- **One recipe, one set of weights per participant.** The architecture, preprocessing and
  hyperparameters are shared; each participant's model is trained only on that participant's
  trials. A recipe is judged by its average over the 32 participants, never tuned per participant.
- **The stated split is within-subject by trial.** A stratified 5-fold over a participant's 40
  trials with one fixed seed; every window of a trial stays on one side of the split.
- **The headline is balanced accuracy for each axis, scored per window**, averaged within a
  participant and then over participants, with the standard deviation across participants. The
  mean of the arousal and valence figures is the single number for comparing runs. Chance is 0.50
  for all three.
- **High means a self-rating at or above 5**, the same line for every participant.
- **The app shows only held-out predictions.** A segment's stored scores come from a model that
  did not train on that segment's trial, so a session looks as good as the headline and no better.
- **The cross-subject result is quoted, not built.** "At chance without calibration" comes from
  the baseline script, kept in the repo as a standalone file. The training code has no
  leave-one-subject-out mode.

## Considered options

| Option | Why not |
| --- | --- |
| One model for all participants, leave-one-subject-out as the stated split | The honest number is 0.50. It matches the literature and ships one checkpoint, but the app would be showing noise. |
| A pretrained foundation model to make cross-subject work | Frozen CBraMod, LaBraM and REVE score 0.48 to 0.53 on DEAP, the same as band features with a linear model. No fine-tuned result is published; nothing suggests a jump. |
| One pooled model, tested on held-out trials of participants it trained on | One checkpoint, and close to what the app does with seeded sessions. Never measured, and it blurs the claim: neither "works on a stranger" nor "calibrated to you". |
| Each participant's median as the line between high and low | Balances the classes (about 20 trials each side) but moves balanced accuracy by at most 0.01, needs per-participant state, and "above your own median" is harder to explain than "at or above the midpoint". Under the fixed line the most lopsided participant still has 6 trials in the smaller class, enough for one in every fold. |
| Score each trial by a vote over its windows | A second scoring path for a number the app never shows; the app stores and displays windows. |

## Consequences

- The claim the app makes is "a decoder calibrated to one person", with DEAP's 32 participants as
  32 stand-ins for that person. It needs labelled recordings from whoever it reads.
- Within-subject on DEAP is within one sitting. A decoder trained on one day and used on another
  is a harder problem this number does not cover.
- A checkpoint holds weights for 32 participants, and a full run is 160 small fits (32
  participants by 5 folds) instead of 32 large ones.
- Seeding cannot run one final model over the data it trained on; it uses the held-out
  predictions.
- Comparing many recipes on the same folds and keeping the best drifts the headline upward. Few
  knobs is the guard.
