# 05: The app loads the new checkpoint, and the old classifiers go

**Spec:** `.scratch/ml-rebuild/spec.md` (The assistant, Deleted, Docs and bookkeeping)

**What to build:** The app describes the classifier that actually scored its sessions. At startup it loads the quadrant-classifier checkpoint; asking the assistant about the model returns the real features, split, figures and training recipe. A checkpoint made under a different recipe stops the app at startup. Everything belonging to the two old classifiers is deleted, so one pipeline is left in the repo. The contrastive encoder is untouched.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] The model holder in the sessions slice holds the loaded checkpoint (recipe and metrics); `override_eeg_model` stays the test seam and takes a checkpoint or `None`
- [ ] A missing checkpoint is a startup warning that names `train-classifier` and hides `get_model_info`; a recipe mismatch fails startup
- [ ] `get_model_info` reports a linear layer on 128 band features, one set of models per participant, the within-subject split, the three figures with their spread, and the training recipe; its failure message names `train-classifier`
- [ ] Deleted: both old classifiers and the CBraMod dual-head loader, the old prediction module with its type check and schema version, the Welch band-power function, the five-band table, the `EEG_MODEL_BACKEND` setting, and the tests that covered only those
- [ ] CBraMod remains as the contrastive encoder's backbone; the contrastive window function, the scale factor and the contrastive tests are unchanged and pass
- [ ] The checkpoint download script fetches only `contrastive_best.pt`
- [ ] The sessions capability's tool list no longer says "EEGNet"
- [ ] `AGENTS.md` (the ML kernel, model holder and scripts paragraphs), `README.md`, `DEVELOPMENT.md`, `GLOSSARY.md`, `docs/ROADMAP.md` (the recorded 52% and anything else now untrue) and the testing-seams rule describe what is built; the architecture guide pages are checked for the old classifiers
- [ ] This ticket's PR also resolves `after-port 01`, as the issue tracker describes

**For the user, not the agent** (a real run is never one of the checks):

- Run `train-classifier` and `seed-sessions` on the real DEAP files; record the three figures wherever the docs quote the number, in place of the estimated 0.57
- Delete the two old classifier files from `backend/data/checkpoints/`
- Remove the classifier files from the GitHub Release, or tell the agent to: it is a public change
