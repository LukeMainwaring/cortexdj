# 06: What preprocessing we own

**Type:** grilling
**Status:** open
**Blocked by:** 01

## Question

Given what DEAP's preprocessed files already did, which steps does our pipeline own and which library does each: baseline correction with the 3-second pre-trial (thrown away today), normalization and at what level (window, trial, participant), the model's input window length and overlap against the 4-second stored segment, and whether the model takes band features or raw EEG windows. Does the hand-written scipy in `ml/preprocessing.py` and the pickle reader in `ml/dataset.py` stay, or move to MNE and braindecode helpers?
