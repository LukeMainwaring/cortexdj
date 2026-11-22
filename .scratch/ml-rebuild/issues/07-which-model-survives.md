# 07: Which model survives

**Type:** grilling
**Status:** resolved
**Blocked by:** 05, 06

## Question

Which one classifier pipeline is kept: a linear or small model on band features, a real EEGNet (or similar small braindecode model) on raw windows, or a fine-tuned CBraMod? Judge by the baseline numbers, how long a fold takes on a laptop, and how easy it is to explain. The loser, its checkpoint, and the branch in `ml/predict.py` are deleted; say whether the arousal and valence heads stay one model with two outputs or become two models.

## Answer

Resolved 2026-11-22. Settled by argument and the earlier tickets; no model was run for this ticket.

- **The model:** a linear layer in PyTorch, `nn.Linear(128, 2)`, on the standardized differential-entropy features (ticket 06). One per fold. Its two outputs are the arousal and valence logits, so it is one model with two outputs, not two models.
- **Training:** full-batch AdamW with a class-balanced loss (`BCEWithLogitsLoss` with `pos_weight`), a fixed number of epochs, no early stopping (32 training trials leave nothing to hold out for it). Learning rate, epochs and weight decay are fixed and stated as untuned; the implementation picks values where the training loss has flattened. No search over them.
- **Why linear:** each fit sees 32 trials (480 windows, 15 near-copies a trial); ticket 02 found every model family near chance on trial-safe splits; and the map puts simplicity above a point or two. It is the one candidate that can be explained weight by weight.
- **Why PyTorch, not scikit-learn:** the user wants the training code to be ordinary PyTorch practice, and the app already loads torch. scikit-learn was only ever the baseline script's tool.
- **scikit-learn:** not a direct dependency. The stratified split by trial, the standardization (a mean and a std saved with the weights) and balanced accuracy are a few lines of numpy each. The kept baseline script (ticket 05) is the one file that imports scikit-learn; it works because braindecode installs it.
- **Deleted:** `EEGNetClassifier` (`ml/model.py`), `PretrainedDualHead` and `load_pretrained_dual_head` (`ml/pretrained.py`), the `eegnet_best.pt` and `cbramod_best.pt` checkpoints, the `EEG_MODEL_BACKEND` setting, and the `isinstance` branch in `ml/predict.py`. CBraMod stays only as the contrastive encoder's backbone.
- **Raw windows:** closed for this map. The quick EEGNet run ticket 06 left open was not done.
- **Expected number:** 0.57 (0.60 valence, 0.55 arousal), which is the scikit-learn baseline's figure. No torch model has been run; the first real training run confirms or corrects it.
- **Out of scope, on the roadmap:** bigger models, raw-window architectures and CBraMod fine-tuning are a later effort, after the app works end to end. It starts with a bake-off on Modal (linear, a small MLP on the features, a real EEGNet on raw windows), and a model replaces linear only if it wins by about 0.03 on the single number. Added to `docs/ROADMAP.md`, Phase 4.
- **Passed to ticket 08:** the folds, metric and training loop take any `nn.Module` plus a function that turns a window into the model's input, so a later model is a new class and not a rewrite. Only one model exists in the repo.
- **No ADR:** with the harness taking any module, the model is cheap to swap.

## Comments

- 2026-11-22: ticket 05 resolved: one model per participant, within-subject by trial (ADR 0005). Judge candidates by the mean of arousal and valence balanced accuracy over the 32 participants, against the baseline's 0.57 (0.60 valence, 0.55 arousal). Each fit sees about 32 trials (480 windows) of one participant, and a full run is 160 fits, so how a candidate copes with that little data and how long 160 fits take both count. A pretrained model's case is now 'needs fewer labelled trials to calibrate', not 'works across people'.

- 2026-11-22: ticket 06 resolved. The input is differential-entropy band features (128 a window, standardized on the training fold), so the choice is among models on those features; raw windows reopen only if a quick EEGNet run clearly beats 0.57. Also decide here whether scikit-learn becomes a dependency (scaler, fold splitter, and the model itself if it is linear).
