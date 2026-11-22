# 08: Where training code lives, the checkpoint, and how Modal fits

**Type:** grilling
**Status:** resolved
**Blocked by:** 03, 07

## Question

Where does training code live so the `cortexdj.ml` inference kernel stays import-clean, and how do training and inference share one preprocessing path without copying it? What is in a checkpoint (weights, the preprocessing settings it was trained with, metrics) so a mismatch like `after-port 01` fails loudly at load, and how does a checkpoint get from a run to the app? Which is the default command, the local single-fold run or the Modal full run, and how few settings can the training function take so there is one code path?

## Answer

Resolved 2026-11-22. Settled by argument and the earlier tickets; nothing was run. The one timing behind it is ticket 04's: the scikit-learn baseline did a full run in 40 seconds on a laptop. The torch version is untimed and assumed to be about a minute.

- **Modal:** not in this effort. No wrapper file, no `modal` dependency, no Volume. The user wants Modal as the eventual default training path, so the training function is shaped for it now (below) and the wrapper arrives with the bake-off effort, where ticket 03's research applies unchanged.
- **Unit of training:** the participant, not the fold. `train_participant(participant_id, data_dir, recipe=RECIPE)` extracts features, makes the 5 folds, fits 5 models and returns them with their metrics as plain data. It reads no env and no global state, which is what Modal's `.map` needs later. A fold fit is a small inner function. A full run is a loop over 32 participants. CPU only, no device argument.
- **Where it lives:** `src/cortexdj/training/` (folds, metric, the loop, the command). It imports `ml/` and `core/`. A new import-linter contract says nothing else imports `training`. The kept baseline script (ticket 05) sits there as one standalone file.
- **Command:** `train-classifier` under `[project.scripts]`, the one way to train. One option, `--participants`, like `seed-sessions`.
- **Training recipe:** one frozen dataclass of constants in `ml/` (band edges, window length, sample rate, label threshold, fold count and seed, learning rate, epochs, weight decay). Not env vars or pydantic settings: it is part of the model, not per-machine configuration. The optional `recipe` argument is the one opening for a later autoresearch loop. The seed is in it, so a rerun gives the same number.
- **Shared preprocessing:** one function in `ml/` turns a 4-second window into the 128 features; training and seeding both call it.
- **Standardization:** inside the model: `nn.Sequential(Standardize(mean, std), nn.Linear(128, 2))`, with the mean and std as buffers on a small `Standardize` module. The model's input is unstandardized features and its saved weights carry their own scaling. Turned down: a separate scaler object saved beside the weights (can be forgotten at load), and folding the scaling into the linear weights (works only for a linear model).
- **Checkpoint:** one file, `quadrant_classifier.pt`, about 330KB: per participant the 5 fold models and which trial went to which fold, plus the metrics and the training recipe. It stores no predictions.
- **Seeding:** `seed-sessions` scores each segment with the fold model that did not train on its trial. It refuses to run without a checkpoint and says to run `train-classifier`; the fallback that wrote the participants' ratings as predictions goes. Turned down: training saves the held-out predictions and seeding reads them (stored scores could drift from the code, and nothing would exercise the prediction path).
- **Mismatch at load:** loading compares the checkpoint's stored recipe with the code's and raises naming the first setting that differs. Plain values, not a hash. `CHECKPOINT_SCHEMA_VERSION` goes.
- **From a run to the app:** made locally, never downloaded. Setup is `train-classifier`, then `seed-sessions`. The classifier leaves the GitHub Release; `download-checkpoints.sh` fetches only `contrastive_best.pt`.
- **At startup:** the app keeps the holder in `features/sessions/eeg_model.py`, now holding the loaded checkpoint (recipe and metrics) for `get_model_info`. A missing file stays a warning that hides the tool; a recipe mismatch fails startup.
- **Tests:** a fixture builds one fake participant in DEAP's shape with a planted signal, and the unit tier runs `train_participant` on it in a few seconds. Checked: a trial never straddles a fold, every trial is scored once, save then load gives the same predictions, a changed recipe raises at load. CI never touches DEAP and no real training run is a check.
- **No ADR:** each of these is cheap to change.

**Open on naming:** `GLOSSARY.md` already has "Shared recipe" for the agent build recipe, and ADR 0005 says "one recipe" for training. This answer says "training recipe" to keep them apart; the glossary was not changed.

**Not verified:** the one-minute run time, the 330KB file size (computed from 160 x 514 floats), and that `Standardize` buffers round-trip through `torch.load(weights_only=True)`.

## Comments

- 2026-11-22: ticket 05 resolved: one model per participant, within-subject by trial (ADR 0005). This ticket now also decides: what a checkpoint for 32 participants looks like on disk; that the fan-out is over 32 participants (5 folds each), not the 32 held-out folds ticket 03 costed; whether the run saves the fold models or their held-out predictions, since the app may only show those; and where the standalone baseline script lives (today only on local branch `refs/heads/prototype/ml-rebuild-baseline`, bcfe9dd).

- 2026-11-22: ticket 07 resolved: the model is `nn.Linear(128, 2)` in PyTorch on the band features, full-batch AdamW, fixed settings; scikit-learn is not a dependency, so the trial split, standardization and balanced accuracy are plain numpy. A fold's checkpoint is small: the layer's weights plus the feature mean and std. Constraint from that ticket: the folds, metric and training loop take any `nn.Module` plus a function that turns a window into the model's input, with one model in the repo, so the later bake-off on Modal adds classes and not a second pipeline. Fits are fast enough that a full local run of 160 may be practical; decide the default command with that in mind.
