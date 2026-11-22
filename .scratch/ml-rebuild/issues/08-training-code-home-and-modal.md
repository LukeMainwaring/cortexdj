# 08: Where training code lives, the checkpoint, and how Modal fits

**Type:** grilling
**Status:** open
**Blocked by:** 03, 07

## Question

Where does training code live so the `cortexdj.ml` inference kernel stays import-clean, and how do training and inference share one preprocessing path without copying it? What is in a checkpoint (weights, the preprocessing settings it was trained with, metrics) so a mismatch like `after-port 01` fails loudly at load, and how does a checkpoint get from a run to the app? Which is the default command, the local single-fold run or the Modal full run, and how few settings can the training function take so there is one code path?

## Comments

- 2026-11-22: ticket 05 resolved: one model per participant, within-subject by trial (ADR 0005). This ticket now also decides: what a checkpoint for 32 participants looks like on disk; that the fan-out is over 32 participants (5 folds each), not the 32 held-out folds ticket 03 costed; whether the run saves the fold models or their held-out predictions, since the app may only show those; and where the standalone baseline script lives (today only on local branch `refs/heads/prototype/ml-rebuild-baseline`, bcfe9dd).

- 2026-11-22: ticket 07 resolved: the model is `nn.Linear(128, 2)` in PyTorch on the band features, full-batch AdamW, fixed settings; scikit-learn is not a dependency, so the trial split, standardization and balanced accuracy are plain numpy. A fold's checkpoint is small: the layer's weights plus the feature mean and std. Constraint from that ticket: the folds, metric and training loop take any `nn.Module` plus a function that turns a window into the model's input, with one model in the repo, so the later bake-off on Modal adds classes and not a second pipeline. Fits are fast enough that a full local run of 160 may be practical; decide the default command with that in mind.
