# 04: A baseline run on band features

**Type:** task
**Status:** open

## Question

Run a throwaway baseline so later decisions rest on real numbers: logistic regression on differential-entropy band features from the 32 DEAP files in `backend/data/deap`, for arousal and valence, under leave-one-subject-out and under a within-subject split by trial. Report balanced accuracy and the majority baseline for each, with the fixed threshold at 5 and with each participant's median as the label rule, and how long it took. The code lives on a `prototype/` branch and is not the build.
