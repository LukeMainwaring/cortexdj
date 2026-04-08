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

## Usage

```bash
# Train EEGNet on DEAP
uv run --directory backend train-model --source deap

# Train CBraMod pretrained model on DEAP
uv run --directory backend train-model --source deap --model cbramod --cv loso

# Compare both models
uv run --directory backend compare-models

# Seed database with DEAP sessions
uv run --directory backend seed-sessions --source deap
```
