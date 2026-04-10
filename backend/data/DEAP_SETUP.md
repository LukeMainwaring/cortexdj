# DEAP Dataset Setup

CortexDJ supports training on the [DEAP dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) — a widely cited benchmark for EEG-based emotion recognition (32 participants, 40 music video trials each).

## Download Instructions

1. Download from Kaggle: https://www.kaggle.com/datasets/manh123df/deap-dataset/data
2. Extract and copy the `.dat` files from `data_preprocessed_python/` into this directory:

```
backend/data/deap/
├── s01.dat
├── s02.dat
├── ...
└── s32.dat
```

> **Note:** The official DEAP download page (eecs.qmul.ac.uk) is currently unavailable. The Kaggle mirror above contains the same preprocessed data.

## Verify

```bash
ls backend/data/deap/*.dat | wc -l
# Should output: 32
```

## Data Format

Each `.dat` file is a Python pickle containing:

- `data`: `(40, 40, 8064)` — 40 trials, 40 channels (32 EEG + 8 peripheral), 8064 samples at 128Hz
- `labels`: `(40, 4)` — [valence, arousal, dominance, liking] on 1-9 scale

The first 3 seconds (384 samples) of each trial are baseline; the remaining 60 seconds are the stimulus response.

CortexDJ's loader automatically strips the baseline and extracts the 32 EEG channels.

## Label Binarization

The 1–9 Likert self-reports get binarized to low/high for the dual-head classifier. Three strategies are available via `--label-split`:

- **`median_per_subject`** (default, recommended): each subject is split at their own Likert median per axis, giving balanced classes per subject and removing per-subject rating-scale bias. This is the post-fix default and should be what you use unless you have a specific reason otherwise.
- **`median_global`**: pooled median across all 32 subjects. Slightly less balanced per-fold but deterministic across subjects.
- **`fixed_5`**: legacy `>= 5` threshold — produces a ~24/76 high/low split on DEAP. Only useful for reproducing papers that adopted this convention. **Note:** older training logs (before the collapse fix) used this default and reported ~0.77 accuracy numbers that were dominated by majority-class predictions. If you compare against those numbers, expect the new `median_per_subject` default to show lower raw accuracy but dramatically higher macro-F1 — the new metric is the honest one.

The label split strategy is encoded in the `.npz` cache key, so switching is free after the first build of each strategy.

## Usage

```bash
# Train CBraMod on DEAP with LOSO CV (default — 50 epochs, all 32 folds,
# median_per_subject labels, class-weighted CE with label smoothing)
uv run --directory backend train-model

# Quick dev run (10 epochs, 3 folds) — works on Apple Silicon MPS
uv run --directory backend train-model --quick

# Train EEGNet instead
uv run --directory backend train-model --model eegnet

# Compare both models (always renders a MajorityBaseline reference row
# from dataset labels; a trained model must beat it on macro-F1 or
# something is wrong)
uv run --directory backend compare-models

# Reproduce a DEAP paper that used the historical >= 5 threshold
uv run --directory backend train-model --label-split fixed_5

# Seed database with DEAP sessions
uv run --directory backend seed-sessions
```
