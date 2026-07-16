# Binarize DEAP labels at each subject's own median

**Status**: Accepted — 2026-07-16

DEAP's valence/arousal labels are 1–9 Likert self-reports. The convention in the
DEAP literature is to threshold at `>= 5`, which here produces a ~25/75 class
skew and bakes in per-subject rating-scale bias — some participants never use
the low end of the scale, so the "low arousal" class is partly a fact about the
rater, not the trial. Under our leave-one-subject-out CV regime that bias lands
squarely in the held-out fold.

`ml/dataset.py` therefore defaults to `median_per_subject`: each axis is split
at that subject's own median across their 40 trials, giving roughly balanced
labels per subject and removing the scale bias. `median_global` (pooled median)
and `fixed_5` remain opt-in via `--label-split`; `fixed_5` exists to reproduce
published numbers and should not be read as our baseline.

**Consequence**: accuracy figures here are not directly comparable to papers
using the `>= 5` split — a `fixed_5` run is the apples-to-apples comparison.
