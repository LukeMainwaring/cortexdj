# 02: What recent benchmarks report for DEAP arousal and valence

**Type:** research
**Status:** resolved

## Question

What do recent, trustworthy benchmarks report for binary arousal and valence on DEAP, split by evaluation protocol: cross-subject (leave-one-subject-out), within-subject with trial-level splits, and within-subject with window-level splits (which leak)? For each: what small models and classic features (differential entropy, band power with a linear model, EEGNet) reach, what foundation models (CBraMod, LaBraM and similar, per NeuralBench and EEG-FM-Bench) reach, which metric they use, how they binarize the 1 to 9 ratings (fixed threshold at 5, per-participant median), and what the majority-class baseline is under each. The aim is an honest expectation for a small model and the facts needed to pick a split and a label rule.

## Answer

Full findings, with the table of every number and its source: `docs/research/deap-benchmarks-by-protocol.md` on branch `refs/heads/research/ml-rebuild-deap-benchmarks` (commit d5674d6, pushed, never merged).

- **Honest expectation for a small model:** cross-subject, chance up to about 0.55 balanced accuracy; within-subject with trial-level folds, 0.52 to 0.57. Plain accuracy sits at or below the majority baseline in most published trial-safe results.
- **Cross-subject (leave-one-subject-out, 4s windows):** EEGNet, TSception and similar reach 0.53 to 0.57 arousal and 0.51 to 0.53 valence accuracy, against a majority baseline of 0.59 and 0.57 (EEGain, arXiv 2505.18175). An SVM on differential entropy lands in the same place (LibEER, arXiv 2410.09767).
- **Foundation models do no better:** frozen CBraMod, LaBraM and REVE with a linear probe score 0.48 to 0.53 balanced accuracy on DEAP (OmniEEG-Bench, arXiv 2606.00815). No source reports a fine-tuned foundation model on DEAP; NeuralBench and EEG-FM-Bench do not include it.
- **Within-subject, trial-level splits:** about 0.58 to 0.62 accuracy for SVM and small conv nets (TSception paper, arXiv 2104.02935), against per-participant majority baselines of 0.58 valence and 0.64 arousal. The original DEAP paper's naive Bayes scored below its own majority baseline.
- **Window-level splits leak:** the same model moves from about 52% to over 90% when windows from one trial land on both sides (Khan et al., arXiv 2508.02417; LibEER reproduces 92% falling to 63%). Of 101 DEAP papers Khan et al. reviewed, 58 had this flaw.
- **Label rules and majority baselines** (from the published ratings, valence / arousal): `>= 5` gives 56.6% / 58.9% pooled; `> 4.5` gives 63.1% / 63.8%; per-participant median gives about 51%. Balanced-accuracy chance is 0.5 under every rule.
- **The local labels are not the published ones** (see ticket 09): in exactly the 439 trials where both published ratings exceed 5, the local files hold `9 - rating` for valence and arousal. The excited quadrant is empty locally (re-checked in this session: zero trials with both above 5), against 34.3% in the published ratings. The repo's recorded 52% was scored against these labels.
- **Not verified:** several papers do not state their label rule; the ratings file was authenticated only by reproducing the DEAP paper's statistics; a possible header misprint in OmniEEG-Bench's supplementary tables.
