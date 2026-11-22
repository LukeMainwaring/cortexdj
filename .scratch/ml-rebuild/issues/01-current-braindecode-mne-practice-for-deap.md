# 01: Current braindecode and MNE practice for DEAP

**Type:** research
**Status:** resolved

## Question

With braindecode 2.0 and MNE 1.13, what is the recommended way today to load DEAP's preprocessed `.dat` files, preprocess them, and cut windows for training? Specifically: is there a maintained DEAP loader (braindecode, MOABB, torcheeg, or other) or is wrapping the arrays ourselves the norm; what braindecode's own preprocessing and windowing helpers give over the hand-written scipy in `backend/src/cortexdj/ml/preprocessing.py` and `dataset.py`; what DEAP's own preprocessing already did, so what is left to do (baseline correction with the 3-second pre-trial, normalization, re-referencing); and which braindecode models and training helpers (EEGNet, ShallowFBCSPNet, skorch `EEGClassifier`) are the current small-model defaults.

## Answer

Full findings: `docs/research/2026-11-22-braindecode-mne-deap-pipeline.md` on branch `refs/heads/research/ml-rebuild-braindecode-mne-deap` (commit b294092, pushed, never merged). Checked against the installed packages and the 32 local `.dat` files.

- **No maintained loader to adopt.** braindecode 2.0, MOABB and MNE have no DEAP dataset. torcheeg has `DEAPDataset`, but its last PyPI release is from December 2024 and it was not tested on Python 3.14. Wrapping the arrays ourselves is the norm, and the pickle read stays in every route.
- **Wrapping routes.** Plain numpy `(n_windows, 32, n_times)` goes straight into a `TensorDataset` or `EEGClassifier.fit(X, y)`. `create_from_X_y` types channels as `misc`, so braindecode's `preprocess` helpers fail on it. `mne.EpochsArray` works but wants volts and a channel order that differs from MNE's `biosemi32` montage.
- **What the helpers add.** Windowing helpers add stride, overlap and window-to-trial metadata; for non-overlapping windows a `reshape` is the same. Neither library ships differential entropy.
- **DEAP already did most of it.** 128Hz, eye artifacts removed, 4 to 45Hz band-pass, common average reference, and the 3-second pre-trial mean already subtracted (confirmed: the pre-trial mean is exactly zero), so baseline correction is a no-op. Left for us: pick 32 channels, drop 384 samples, normalize, window, binarize labels.
- **Normalization is the real open choice.** EEG standard deviation runs from 4.4 (s01) to 59.9 (s04) across participants.
- **The delta band is empty.** 1 to 4Hz sits below DEAP's 4Hz high-pass, so the delta feature and the delta band power the app shows measure filter roll-off.
- **Leakage.** Windows from one trial share a label, so any split must group by trial.
- **Small models.** `braindecode.models.EEGNet` (2,130 parameters, about 1.9s per epoch per participant on CPU) and `ShallowFBCSPNet` (54,562, about 0.5s) take `(batch, 32, n_times)` and return logits. Every braindecode model has one output layer, so arousal plus valence means two models, a 4-class target, or custom code. A plain PyTorch loop is about 30 lines in braindecode's own tutorial; `EEGClassifier` is the skorch alternative. Save `state_dict`, not whole-model pickles.
- **Labels look altered** (re-checked in this session): in the local Kaggle-mirror files, valence has 110 of 1280 values below 1 and arousal 53, minimum 0.0, on a documented 1 to 9 scale; medians are 3.08 and 3.32; only 22% and 25% are at or above 5. Dominance and liking stay within 1 to 9. Raised as ticket 09.
- **Not verified:** no model was trained to convergence; the DEAP readme was read from a Wayback capture; torcheeg untested.
