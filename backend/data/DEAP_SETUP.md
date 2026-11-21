# DEAP Dataset Setup

CortexDJ's EEG sessions come from the [DEAP dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) — a widely cited benchmark for EEG-based emotion recognition (32 participants, 40 music video trials each). `seed-sessions` reads the 32 files, classifies each segment with the downloaded checkpoint (`./backend/scripts/download-checkpoints.sh`), and writes one session per participant; retrieval re-reads a participant's file when it embeds a session.

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

The 1–9 Likert self-reports get binarized to low/high for the dual-head classifier. The dataset code (`ml/dataset.py`, `label_split_strategy`) has three strategies:

- **`median_per_subject`** (default, recommended): each subject is split at their own Likert median per axis, giving balanced classes per subject and removing per-subject rating-scale bias.
- **`median_global`**: pooled median across all 32 subjects. Slightly less balanced per-fold but deterministic across subjects.
- **`fixed_5`**: `>= 5` threshold — produces a ~24/76 high/low split on DEAP. Only useful for reproducing papers that adopted this convention.

The label split strategy is encoded in the `.npz` cache key, so switching is free after the first build.

## Seed the sessions

```bash
uv run --directory backend seed-sessions
```

`DEVELOPMENT.md` says what the command needs and how long it takes.
