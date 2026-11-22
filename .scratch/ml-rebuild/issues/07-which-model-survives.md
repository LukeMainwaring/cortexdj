# 07: Which model survives

**Type:** grilling
**Status:** open
**Blocked by:** 05, 06

## Question

Which one classifier pipeline is kept: a linear or small model on band features, a real EEGNet (or similar small braindecode model) on raw windows, or a fine-tuned CBraMod? Judge by the baseline numbers, how long a fold takes on a laptop, and how easy it is to explain. The loser, its checkpoint, and the branch in `ml/predict.py` are deleted; say whether the arousal and valence heads stay one model with two outputs or become two models.

## Comments

- 2026-11-22: ticket 05 resolved: one model per participant, within-subject by trial (ADR 0005). Judge candidates by the mean of arousal and valence balanced accuracy over the 32 participants, against the baseline's 0.57 (0.60 valence, 0.55 arousal). Each fit sees about 32 trials (480 windows) of one participant, and a full run is 160 fits, so how a candidate copes with that little data and how long 160 fits take both count. A pretrained model's case is now 'needs fewer labelled trials to calibrate', not 'works across people'.
