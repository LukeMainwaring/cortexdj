# 06: What preprocessing we own

**Type:** grilling
**Status:** resolved
**Blocked by:** 01, 10

## Question

Given what DEAP's preprocessed files already did, which steps does our pipeline own and which library does each: baseline correction with the 3-second pre-trial (thrown away today), normalization and at what level (window, trial, participant), the model's input window length and overlap against the 4-second stored segment, and whether the model takes band features or raw EEG windows. Does the hand-written scipy in `ml/preprocessing.py` and the pickle reader in `ml/dataset.py` stay, or move to MNE and braindecode helpers?

## Answer

Resolved 2026-11-22. The library and feature evidence is in ticket 10.

- **Model input:** differential-entropy band features, not raw EEG windows. Ticket 07 picks what sits on top of them, and reopens raw windows only if a quick EEGNet run clearly beats 0.57.
- **Window:** 4 seconds, no overlap, 15 a trial. The model's window and the stored Segment are the same slice of EEG, so a segment's held-out prediction is the fold model's output for that window. Window length is a commonly tuned knob in the field; it stays untuned here.
- **Pre-trial:** the first 384 samples are dropped and nothing else is done with them. DEAP already subtracted the pre-trial mean.
- **Normalization:** one per-feature standardization, fitted on each fold's training trials and saved with that fold's model. Nothing is done to the raw signal, and nothing per window or per trial. The scale gap between participants does not matter because no model sees two participants.
- **Feature:** a Butterworth band-pass then 0.5 x log(variance), filtered per 4-second window, for each of 32 channels and 4 bands: 128 features a window.
- **Bands:** theta 4 to 8, alpha 8 to 13, beta 13 to 30, gamma 30 to 45Hz. No delta: DEAP's 4Hz high-pass removed it. These replace today's `FREQ_BANDS` for the model (delta 1 to 4, alpha to 14, gamma to 40). What the app displays, delta included, is ticket 11.
- **Library:** numpy and scipy, hand-written; the pickle reader stays and a `reshape` does the windowing. No MNE, braindecode helper, or other signal library. The filter moves to scipy's second-order-sections form (`sosfiltfilt`). The docstring in `ml/preprocessing.py` that names MNE is wrong and goes. Whether scikit-learn joins the dependencies (scaler, fold splitter) is left to ticket 07.
- **Why not MNE:** its strengths are reading raw recordings, filtering, artifact removal and epoching, all of which DEAP's authors did before publishing; it stops at the spectrum, and its objects want volts and a channel layout DEAP's order does not match. It is installed only because braindecode depends on it.
- **Alternative noted, not taken:** pyRiemann covariance and tangent-space features, if a second feature family is ever wanted.
- **App contract:** not reopened. Neither this ticket nor ticket 05 found the 4-second segment or the 0 to 1 scores wrong; the displayed delta band is the one open piece (ticket 11).
- **Unchanged:** the contrastive path (`trial_to_eeg_windows`, `CBRAMOD_SCALE_FACTOR`).
- **Not covered:** no head-to-head of feature sets on our split (ticket 10's limits). With a Muse headband and the user's own recordings, filtering and artifact removal become ours and MNE is the natural tool; that stays out of scope.

## Comments

- 2026-11-22: the baseline (ticket 04) found per-participant z-scoring of the features is the one thing that moved leave-one-subject-out off chance (0.50 to about 0.53 balanced accuracy, logistic regression on differential entropy). It needs a participant's whole recording before any window is scored, which matters for how the app's sessions are classified.
- 2026-11-22: ticket 05 resolved: one model per participant, within-subject by trial (ADR 0005). Each model only ever sees one participant, so normalization at the participant level is the natural default; fit it on the training trials of a fold so the held-out trials stay unseen.
