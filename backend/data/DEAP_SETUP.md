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
├── s32.dat
└── participant_ratings.csv
```

3. Download DEAP's published ratings, `participant_ratings.csv`, into the same directory. The official page does not serve it today; a public copy is in the `data/` folder of the GitHub repo `tomdamelio/arousal_decoding` (`master`), and `hsiehjackson/Emotion-Recognition-DEAP_GSR` (`zip/`) holds a byte-identical one. Check it:

```bash
shasum -a 256 backend/data/deap/participant_ratings.csv
# 7bc0c19d31769b01597866ca55b59221635a7bf4237a27c820ef437e1d8ea1b8  (47,969 bytes, 1,280 rows)
```

The directory is gitignored: DEAP's licence forbids redistributing either file, so never commit them.

> **Note:** The official DEAP download page (eecs.qmul.ac.uk) is currently unavailable. The Kaggle mirror's valence and arousal are altered: in the 439 trials where both published ratings exceed 5, the mirror stores `9 - rating`. Its dominance and liking match the published ratings. CortexDJ trusts only the mirror's EEG array and takes valence and arousal from `participant_ratings.csv`. The EEG has not been compared with an official copy.

## Verify

```bash
ls backend/data/deap/*.dat | wc -l
# Should output: 32
ls backend/data/deap/participant_ratings.csv
```

## Data Format

Each `.dat` file is a Python pickle containing:

- `data`: `(40, 40, 8064)` — 40 trials, 40 channels (32 EEG + 8 peripheral), 8064 samples at 128Hz
- `labels`: `(40, 4)` — [valence, arousal, dominance, liking] on 1-9 scale. In the mirror's copy valence and arousal are altered (see above), so CortexDJ reads them from the ratings file and uses this array only to check alignment.

`participant_ratings.csv` has one row per participant and trial, in presentation order. It joins to a `.dat` file on `Participant_id` and `Experiment_id`: trial index `e` in `sNN.dat` is `Experiment_id` `e + 1`. Loading a participant's ratings fails if their dominance and liking differ from the `.dat` file's own.

The first 3 seconds (384 samples) of each trial are baseline; the remaining 60 seconds are the stimulus response.

CortexDJ's loader automatically strips the baseline and extracts the 32 EEG channels.

## Label Binarization

A rating is high at or above 5 on the 1–9 scale, the same line for every participant. On the published ratings that makes about 57% of trials high on valence and 59% on arousal.

The shipped checkpoints predate this: they were trained on the mirror's altered labels, binarized at each subject's own median per axis.

## Seed the sessions

```bash
uv run --directory backend seed-sessions
```

`DEVELOPMENT.md` says what the command needs and how long it takes.
