# 01: Every seeded segment is classified calm

**Status:** needs-planning
**Planning approach:** grill-with-docs

**Planning gate:** Refine against the current implementation with the user before
implementation. Update scope, acceptance criteria, and dependencies; record
settled domain terms and architectural decisions where needed. Return to
ready-for-agent once the user agrees the plan is settled.

**What is wrong:** `seed-sessions` with the CBraMod checkpoint scores every segment at about
0.49 arousal and 0.49 valence, so all 32 sessions read "Calm & focused throughout".
`predict_segment` in `cortexdj/ml/predict.py` multiplies the EEG by
`CBRAMOD_SCALE_FACTOR` (0.01) before the model sees it, and the release checkpoint
`cbramod_best.pt` was trained without that scaling, so its inputs are 100 times smaller than
the ones it learned on.

**Evidence:** scoring participant 1 with the same checkpoint and no scale factor gives
(0.509, 0.5204), (0.5102, 0.5246), (0.5094, 0.5216) for the first three segments and 599
excited, 1 calm over the session. With the scale factor it gives 600 calm.

**What a fix has to weigh:** the scale factor is the documented CBraMod recipe, so the right
fix is a checkpoint trained with it, and training is not in this repo's package. Dropping the
factor at inference makes the two agree again but keeps a checkpoint fine-tuned off the
pretrained model's input range. Either way the checkpoint's recorded accuracy is about 52%
under leave-one-subject-out, which is chance for a two-way split, so judge a new checkpoint by
balanced accuracy or macro-F1 against the majority baseline, not by how varied the sessions
look.

**What it touches:** the session list, `analyze_session`, `compare_sessions`,
`find_relaxing_tracks`, and `build_mood_playlist`, which all read the stored states. The
sessions need re-seeding after a fix.
