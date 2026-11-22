# 02: `train-classifier` trains every participant and writes the checkpoint

**Spec:** `.scratch/ml-rebuild/spec.md` (Features, Model and training recipe, Split and metric, Training package and command, Checkpoint)

**What to build:** One command, `train-classifier`, trains the quadrant classifier for each participant on their own trials, prints balanced accuracy for arousal, for valence and their mean with the spread across participants, and writes one checkpoint file. The checkpoint can be loaded back, gives the same predictions, and refuses to load under a different training recipe. The new code sits beside the old classifiers; nothing old is changed or deleted here, and the app does not use the checkpoint yet.

**Blocked by:** 01

**Status:** ready-for-agent

- [x] The training recipe is one frozen dataclass of constants in the ML kernel: band edges (theta 4 to 8, alpha 8 to 13, beta 13 to 30, gamma 30 to 45Hz), window length, sample rate, label threshold, fold count, seed, learning rate, epochs, weight decay. Its docstring says the settings are fixed and untuned
- [x] One function in the ML kernel turns a 4-second window into 128 differential-entropy features (second-order-sections Butterworth band-pass, then half the log of the variance, 32 channels by 4 bands), and the same path gives the four displayed band powers (the variance, averaged over channels)
- [x] The model is a linear layer with two outputs behind a standardization module that holds the training fold's mean and standard deviation as buffers
- [x] `train_participant(participant_id, data_dir, recipe=RECIPE)` returns five fold models, which trial went to which fold, and the metrics, as plain data; it reads no environment and no global state, and runs on CPU
- [x] One fold assignment per participant, stratified on the quadrant of the trial's ratings; a trial's windows never straddle a fold and every trial is held out exactly once
- [x] Training raises only when a fold's training trials lack a high or a low on either axis
- [x] Balanced accuracy per axis is scored over a participant's pooled held-out windows, in plain numpy; scikit-learn is not added as a dependency
- [x] Weight decay is one fixed, strong value picked once so few held-out scores fall below 0.1 or above 0.9; no search
- [x] The folds, metric and loop take any torch module plus a function from a window to its input
- [x] `train-classifier` takes `--participants` (default all 32), prints the three figures with their spread, and writes the checkpoint; a subset run overwrites the file and holds only those participants
- [x] The checkpoint holds, per participant, the fold models and fold assignment, plus the metrics and the training recipe; it stores no predictions and loads with weights-only loading
- [x] Loading compares the stored recipe with the code's value by value and raises naming the first setting that differs; a missing file raises a not-found error that names `train-classifier`
- [x] Reading and writing the checkpoint's layout live together in the ML kernel
- [x] The training code is a new `cortexdj.training` package that imports only the ML kernel and the platform kernel, and a new import-linter contract says nothing imports it
- [x] Unit tests run `train_participant` on the made-up participant in a few seconds: a planted signal scores well above chance, shuffled labels score near 0.50, a rerun gives the same metrics, save then load gives the same predictions, a changed recipe raises at load
- [x] `AGENTS.md` and `DEVELOPMENT.md` describe the training package, the command and the import rule; `GLOSSARY.md` gains **Training recipe**, kept apart from Shared recipe
