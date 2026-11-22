# 09: Are the local DEAP labels the official ones?

**Type:** task
**Status:** resolved

## Question

The local files come from a Kaggle mirror (`backend/data/DEAP_SETUP.md`). Their valence and arousal columns have values below 1 (110 and 53 of 1280, minimum 0.0) on a documented 1 to 9 scale, medians near 3.1 and 3.3, and only about a quarter at or above 5; dominance and liking stay within 1 to 9. Compare against an official copy of `data_preprocessed_python` (it needs the signed DEAP licence request, so the user fetches it) or the official `participant_ratings` file: are the labels the same, and is the EEG the same? Record which copy the effort uses from here and where it came from. If the mirror's labels are altered, every number computed on them so far is suspect, the baseline in ticket 04 included.

## Answer

Resolved 2026-11-22. No, the local labels are not DEAP's, and the mirror changed them, not this machine. From here the effort keeps the local EEG files and takes every label from `participant_ratings.csv`.

**What the effort uses**

- **EEG:** the 32 files in `backend/data/deap/` from the Kaggle mirror `manh123df/deap-dataset`, unchanged. Only the `data` array is trusted.
- **Labels:** `backend/data/deap/participant_ratings.csv` (gitignored with the rest of that directory; sha256 `7bc0c19d31769b01597866ca55b59221635a7bf4237a27c820ef437e1d8ea1b8`, 47,969 bytes, 1,280 rows), joined on `Participant_id` and `Experiment_id`. Trial index `e` in `sNN.dat` is `Experiment_id` `e + 1`. Fetched from `tomdamelio/arousal_decoding` (`master`, `data/`); `hsiehjackson/Emotion-Recognition-DEAP_GSR` (`master`, `zip/`) is byte-identical. The file is not committed: the repo is public and DEAP's licence forbids redistribution.
- The `labels` array in the `.dat` files is ignored for valence and arousal. Its dominance and liking columns match the ratings file in all 1,280 trials, which makes them a cheap check that the join is aligned.

**What was checked**

- Re-ran the comparison in this session: valence and arousal differ in 439 trials, exactly the 439 where both published ratings exceed 5, and in each the local value is `9 - published`. High share under `>= 5` is 22.3% / 24.6% locally against 56.6% / 58.9% published.
- All 32 local files are Python 3 pickles (protocol 4, `str` keys, `labels` before `data`), last written 2024-12-11. DEAP's own files are Python 2 pickles from 2012, and the repo's first commit is 2026-03-30, so the mirror's uploader re-saved the files and the change happened upstream.
- The EEG has the properties of the official preprocessing: `(40, 40, 8064)` float64, no NaNs, the mean across the 32 EEG channels is zero at every sample (average reference), and each trial's 3-second baseline averages to zero.

**Not verified**

- The EEG was never compared with an official copy. The user accepted this. Re-check if an official copy turns up.
- The ratings file is authenticated by two identical public copies and by reproducing the DEAP paper's statistics (ticket 02), not by an official download.
- The official site (`www.eecs.qmul.ac.uk/mmv/datasets/deap/`) answered 503 over http and failed the TLS handshake over https on 2026-11-22. A quick web scan found no other host for `data_preprocessed_python` (nothing on Hugging Face, Zenodo, figshare or IEEE DataPort) and no public report of this mirror's labels being altered. The same Kaggle upload lists a `metadata_xls` folder, which may hold the ratings as a third copy; it was not fetched.

**What follows**

- Every number scored on the local labels is void: the recorded 52% leave-one-subject-out figure, and anything from the old training code. The baseline run (ticket 04) must use the ratings file.
- `backend/data/DEAP_SETUP.md` is wrong in two places: "the Kaggle mirror contains the same preprocessed data", and the "~24/76" split it attributes to DEAP. It also needs the ratings-file download step. `ml/dataset.py` reads labels from the `.dat` files today.

## Comments

- 2026-11-22: the benchmarks research (ticket 02) compared the local files with `participant_ratings.csv` from two byte-identical public GitHub mirrors (`tomdamelio/arousal_decoding`, `hsiehjackson/Emotion-Recognition-DEAP_GSR`). In exactly the 439 trials where both published ratings exceed 5, the local valence and arousal are `9 - rating`; dominance and liking match in all 1,280 trials. Local and published labels agree on 65.7% of trials under `>= 5`. The EEG arrays were not compared. So the open parts are: confirm against an official source, check the EEG, and decide where labels come from (the official files, or the ratings file joined on participant and `Experiment_id`). `backend/data/DEAP_SETUP.md`'s "~24/76 split" describes this artefact, not DEAP.
