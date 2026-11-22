# 12: What a re-seeded session shows at about 0.57

**Type:** grilling
**Status:** resolved
**Blocked by:** 08

## Question

After `train-classifier` and a re-seed, every segment's arousal and valence scores are a fold model's held-out output (ticket 08), from a linear layer that is right about 57% of the time. Are the stored 0 to 1 scores the sigmoid of the logits as they come, or do they need calibrating so they are not all bunched near 0.5 or pushed to the ends? What should a session's quadrant path and trajectory analytics look like in the app at that accuracy, and does anything the app or the assistant says about a session need to state how reliable it is? Does the app contract (scores 0 to 1, a quadrant per segment) hold as it is?

## Comments

- 2026-11-22: graduated from the map's fog when ticket 08 settled how seeding gets held-out predictions. No model has been trained yet, so the spread of real scores is unknown; a first `train-classifier` run, or the baseline on `refs/heads/prototype/ml-rebuild-baseline`, would show it.

## Answer

Resolved 2026-11-22. Two throwaway scripts gave the numbers: `.scratch/ml-rebuild/evidence/score_spread.py` and `track_level.py` (same features, labels and folds as ticket 04, keeping the held-out probabilities; run with `uv run --directory backend python <path> data/deap`). They used scikit-learn logistic regression and a quick torch linear layer, not the real `train-classifier`, which does not exist yet.

**What the scores look like**

| | light regularization (`C=1`, or torch at weight decay 0.01) | heavy (`C=0.01`) |
|---|---|---|
| balanced accuracy, valence / arousal | 0.60 / 0.55 | 0.61 / 0.56 |
| segments scoring below 0.1 or above 0.9 | about 30% (50% at 2000 epochs) | 2 to 5% |
| a score above 0.8 is truly high | about 72% | 77 to 85% |
| a segment's quadrant is right (chance 0.25) | 0.37 | 0.38 |
| a track's quadrant is right, from the mean of its 15 segments | 0.43 | 0.43 |
| quadrant changes per session, by segment (599 steps) | about 310, 285 of them inside a track | about 270 |
| quadrant changes per session, by track (39 steps) | 25 | 26 |

The participants' own ratings give 20 changes over the 39 track steps, and by ratings 31 of 32 sessions have no quadrant above 60% of their tracks. Averaging a track's 15 segments lifts balanced accuracy to 0.64 valence and 0.58 arousal.

**Decisions**

- **Scores:** the stored score is the sigmoid of the logit as it comes. No calibration step. A quadrant is still cut at 0.5 on each score. The app contract holds: no migration.
- **Weight decay:** this amends ticket 07. "Train until the loss has flattened" is what pushes scores to the ends, so the training recipe carries one fixed, strong weight decay, stated as untuned. The implementation picks the value once; it is not searched.
- **The track is the unit the app talks in:** time in each quadrant, transitions, the dominant quadrant, centroid, dispersion, path length and the card label are computed over a session's 40 tracks (each track's mean scores, which `SessionTrack` already stores), not its 600 segments. Segments stay stored and charted as they are.
- **Trajectory panel:** backend only. The path is the 40 track points and the three-segment rolling mean goes. The frontend is untouched and keeps drawing the 600 segment dots behind the path.
- **Card labels:** the cut-offs stay (90% "throughout", 60% "mostly", otherwise "mixed"), counted over tracks. Nearly every DEAP card reads "Mixed, leaning …", which is what the ratings give too.
- **Saying how reliable it is:** the sessions capability's instructions carry the loaded checkpoint's single number (the mean over all 32 participants, not each participant's own) and one wording tier from it. Below 0.60: "a rough read". From 0.60 to 0.75: "a fair read". At 0.75 and above: no hedge. One short clause, once per analysis, never a paragraph. The cut-offs are picked, not sourced. Nothing changes in the frontend. The user is fine with it overstating slightly; the wording loosens as the model improves.
- **Canned explanations:** the four paragraphs in `features/sessions/emotion.py` are rewritten to say what a quadrant means, without claiming what the frequency bands did. They are fixed text keyed on the quadrant, and nothing measured the bands for that session. The same goes for the band claims in the system prompt's quadrant list.
- **Muse:** nothing here is tied to DEAP's 32 channels. A Muse model would read 16 features and need the user's own rated sessions; out of scope as before.
- **No ADR:** each of these is cheap to change.

Also written: the glossary's Trajectory entry now says the path goes track by track.

**Not verified:** the real torch model's spread at whatever weight decay is picked; how the panel looks with a 40-point path over 600 dots; whether the assistant keeps to one clause.
