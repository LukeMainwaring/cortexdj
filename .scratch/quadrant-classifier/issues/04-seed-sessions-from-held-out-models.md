# 04: `seed-sessions` scores segments with held-out models and replaces a session

**Spec:** `.scratch/ml-rebuild/spec.md` (Seeding, Features)

**What to build:** After `train-classifier`, running `seed-sessions` gives each participant one session whose every segment was scored by the fold model that did not train on that segment's trial, so a session looks as good as the headline number and no better. Band powers on a segment are the model's own four bands. Running it again replaces the session instead of adding a second. Without a checkpoint it refuses and says what to run. The old classifiers and the app's model holder are not touched here; ticket 05 switches the app.

**Blocked by:** 02

**Status:** ready-for-agent

- [ ] Each segment's arousal and valence scores are the sigmoid of the logits from the fold model that held out its trial, with no calibration; the quadrant is cut at 0.5 on each score
- [ ] A segment's band powers are the four bands from the model's feature path; delta is not computed or stored
- [ ] With no checkpoint, or for a participant the checkpoint does not hold, seeding fails with a message that names `train-classifier` and writes nothing
- [ ] The fallback that stored participants' ratings as predictions is gone, and so is the `--model` option
- [ ] A participant's existing DEAP session is deleted and the fresh one inserted, so seeding twice leaves one session for that participant; tracks stay
- [ ] Any thread's brain context that pointed at the replaced session has its session id cleared in the same transaction; its mood and averages stay
- [ ] The seed function takes the data directory and the checkpoint as arguments, defaulting to the standard paths
- [ ] Integration tests seed the made-up participant with a checkpoint trained on it in the test and check the scores against the held-out models, the replace, the brain-context clear, and both refusals
- [ ] Test factories and any fixture that builds a segment lose delta
- [ ] `README.md`, `DEVELOPMENT.md` and `AGENTS.md` say setup is `train-classifier` then `seed-sessions`, and that a re-seed replaces
