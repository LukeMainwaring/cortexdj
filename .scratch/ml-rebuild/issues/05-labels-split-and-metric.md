# 05: Labels, split and metric

**Type:** grilling
**Status:** resolved
**Blocked by:** 02, 04, 09

## Question

How are the 1 to 9 self-ratings turned into high and low (fixed threshold at 5, or each participant's median), which split is the stated one (cross-subject, or within-subject by trial), and which single metric is the headline number? The choice has to serve two things at once: an honest claim about what the model can do, and sessions in the app that do not all read the same.

## Answer

Resolved 2026-11-22. The decision and the options turned down are in `docs/adr/0005-one-quadrant-classifier-per-participant.md`.

The app is reframed as a decoder calibrated to one person, with DEAP's 32 participants as stand-ins for that person. Cross-subject is chance for every model family (tickets 02 and 04), so one model for everyone has nothing honest to show.

- **Split:** within-subject by trial. One model per participant, trained only on that participant's trials; one recipe shared by all 32, never tuned per participant.
- **Folds:** stratified 5-fold over a participant's 40 trials, one fixed seed, a trial's 15 windows always on one side. Every trial is scored once.
- **Label rule:** high is a self-rating `>= 5`, from `participant_ratings.csv` (ticket 09). Checked for this split: the most lopsided participant has 6 trials in the smaller class for arousal and 10 for valence, so every test fold gets at least one of each. The median rule would give 16 to 20.
- **Metric:** balanced accuracy for each axis, per window, averaged within a participant then over the 32, with the sd across participants. The single number for comparing runs is the mean of the arousal and valence figures. Chance is 0.50 for all three. No trial vote.
- **What the app shows:** a segment's stored scores come only from a model that did not train on that segment's trial (the fold models' held-out predictions). How seeding gets them is still in the map's fog.
- **Cross-subject:** quoted as "at chance without calibration" (0.50, about 0.53 with per-participant z-scoring) from the baseline run. The baseline script is kept in the repo as one standalone file; the training code gets no leave-one-subject-out mode. The script is today only on the local branch `refs/heads/prototype/ml-rebuild-baseline` (bcfe9dd).

**Expectation to beat:** the baseline's 0.60 valence, 0.55 arousal, so 0.57 on the single number.

**Not covered:** DEAP's within-subject is one sitting; a decoder used on a different day from its training is untested. A Muse headband and the user's own rated sessions are where this heads, and stay out of scope.

Also written: the glossary's Quadrant entry now says where high starts.
