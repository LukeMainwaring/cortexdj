# 07: Which model survives

**Type:** grilling
**Status:** open
**Blocked by:** 05, 06

## Question

Which one classifier pipeline is kept: a linear or small model on band features, a real EEGNet (or similar small braindecode model) on raw windows, or a fine-tuned CBraMod? Judge by the baseline numbers, how long a fold takes on a laptop, and how easy it is to explain. The loser, its checkpoint, and the branch in `ml/predict.py` are deleted; say whether the arousal and valence heads stay one model with two outputs or become two models.
