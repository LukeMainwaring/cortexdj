# 09: Are the local DEAP labels the official ones?

**Type:** task
**Status:** open

## Question

The local files come from a Kaggle mirror (`backend/data/DEAP_SETUP.md`). Their valence and arousal columns have values below 1 (110 and 53 of 1280, minimum 0.0) on a documented 1 to 9 scale, medians near 3.1 and 3.3, and only about a quarter at or above 5; dominance and liking stay within 1 to 9. Compare against an official copy of `data_preprocessed_python` (it needs the signed DEAP licence request, so the user fetches it) or the official `participant_ratings` file: are the labels the same, and is the EEG the same? Record which copy the effort uses from here and where it came from. If the mirror's labels are altered, every number computed on them so far is suspect, the baseline in ticket 04 included.

## Comments

- 2026-11-22: the benchmarks research (ticket 02) compared the local files with `participant_ratings.csv` from two byte-identical public GitHub mirrors (`tomdamelio/arousal_decoding`, `hsiehjackson/Emotion-Recognition-DEAP_GSR`). In exactly the 439 trials where both published ratings exceed 5, the local valence and arousal are `9 - rating`; dominance and liking match in all 1,280 trials. Local and published labels agree on 65.7% of trials under `>= 5`. The EEG arrays were not compared. So the open parts are: confirm against an official source, check the EEG, and decide where labels come from (the official files, or the ratings file joined on participant and `Experiment_id`). `backend/data/DEAP_SETUP.md`'s "~24/76 split" describes this artefact, not DEAP.
