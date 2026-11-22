# Spec: Rebuild the signal pipeline and the quadrant classifier

**Status:** ready-for-agent

Settled by the wayfinder map in this directory (`map.md`) and its thirteen decision tickets (`issues/01` to `13`). Where this spec and a ticket disagree, the ticket's `## Answer` wins, except for the calls listed under Further Notes, which the spec made because no ticket did. The architectural decision is `docs/adr/0005-one-quadrant-classifier-per-participant.md`.

Implementation tickets: `.scratch/quadrant-classifier/issues/` (01 to 08). They live in their own effort because this directory's `issues/` holds the map's decision tickets.

## Problem Statement

Every seeded session in the app reads "Calm & focused throughout". The classifier checkpoint the app downloads was trained without the input scaling that inference applies, so all 600 segments of every session score about 0.49 on both axes (`after-port 01`).

Fixing the scaling would not make the result worth showing:

- The labels the checkpoint was trained on are not DEAP's. The Kaggle mirror the EEG files came from changed valence and arousal in 439 of 1,280 trials, so the recorded 52% figure, and every number from the old training code, was scored against wrong labels.
- The claim behind it, one model that reads anyone's brain, is chance on DEAP for every model family, foundation models included.
- There are two classifier pipelines (a small feature network and a fine-tuned CBraMod), a setting to pick between them, and a type check at inference to tell them apart. The training code for both was removed from the repo, so neither can be retrained, explained end to end, or checked.
- The band powers the session chart draws come from a second computation with different band edges from the model's, and its delta band measures the roll-off of DEAP's own 4Hz filter.
- The assistant's canned explanations say what a listener's frequency bands did, though nothing measured them for that session.

The user cannot explain how a segment gets its quadrant, cannot reproduce the model, and has no honest number to put next to it.

## Solution

One classifier pipeline, small enough to explain weight by weight, with its training code in the repo and one honest number beside it.

The app becomes a decoder calibrated to one person, with DEAP's 32 participants standing in for that person. One training recipe is shared by all; each participant gets their own models, trained only on their own trials and scored on trials those models never saw. The model is a single linear layer on differential entropy in four frequency bands. The headline is balanced accuracy per axis on a stated split, expected near 0.57 against a chance of 0.50, and the app says so in a short hedge instead of hiding it.

Setup becomes two commands: `train-classifier` (about a minute on a laptop, makes the checkpoint locally) and `seed-sessions` (scores each segment with a model that did not train on that segment's trial, and replaces a participant's existing session). After a re-seed, sessions differ from one another, the chart shows the four bands the model reads, the trajectory and the card labels are counted per track, and the assistant calls its reading "a rough read".

## User Stories

1. As the developer, I want one classifier pipeline in the repo, so that there is one thing to read, explain and maintain.
2. As the developer, I want to explain how a segment gets its quadrant from raw EEG to stored score, so that I can defend every step.
3. As the developer, I want the training code in the repo beside the inference code, so that the checkpoint the app loads can be reproduced.
4. As the developer, I want one command that trains every participant's models in about a minute on a laptop, so that I never need a GPU, a cloud account or a long wait to get a checkpoint.
5. As the developer, I want to train a subset of participants with `--participants`, so that a quick check takes seconds.
6. As the developer, I want the training run to print balanced accuracy for arousal, for valence, and their mean, each with its spread across participants, so that I can state the model's quality in one line.
7. As the developer, I want the same command to give the same number on a rerun, so that a change in the number means a change in the code.
8. As the developer, I want labels to come from DEAP's published ratings and never from the altered arrays in the mirror's files, so that the number is scored against what participants actually reported.
9. As the developer, I want loading to fail if the ratings file is missing or does not line up with the EEG files, so that a misaligned join cannot silently produce a number.
10. As the developer, I want every window of a trial kept on one side of a split, so that the number is not inflated by near-copies leaking across it.
11. As the developer, I want the model's input window and the stored segment to be the same slice of EEG, so that a segment's score is exactly one model output.
12. As the developer, I want training and seeding to compute features through one function, so that they cannot drift apart.
13. As the developer, I want each model to carry its own feature scaling, so that scaling cannot be forgotten or mismatched at load.
14. As the developer, I want the checkpoint to carry the training recipe it was made with, so that the file says how it was produced.
15. As the developer, I want loading a checkpoint made with a different recipe to fail and name the setting that differs, so that a mismatch like `after-port 01` cannot reach the database.
16. As the developer, I want the app to refuse to start on a mismatched checkpoint, so that a stale file is found at startup and not in a chat.
17. As the developer, I want the app to start without a checkpoint, with a warning, so that the rest of the app works before I have trained anything.
18. As the developer, I want `seed-sessions` to refuse to run without a checkpoint and tell me to run `train-classifier`, so that participants' own ratings are never written to the database dressed as predictions.
19. As the developer, I want `seed-sessions` to score each segment with the model that did not train on that segment's trial, so that a session looks as good as the headline number and no better.
20. As the developer, I want a re-seed to replace a participant's session instead of adding a second, so that I can re-seed after retraining without resetting the database or restoring the track index.
21. As the developer, I want a thread's brain context cleared of a session that a re-seed replaced, so that the assistant is never told to focus on a session that no longer exists.
22. As the developer, I want the cross-subject result kept as one standalone script, so that "at chance without calibration" can be re-run and not just quoted.
23. As the developer, I want `train_participant` to be a plain function of a participant id, a data directory and a recipe, so that a later Modal wrapper or a recipe search can call it without reshaping it.
24. As the developer, I want an import rule that nothing in the app imports the training package, so that the inference kernel stays free of training code.
25. As the developer, I want the old classifiers, their checkpoints, the backend setting and the type check at inference deleted, so that no dead path is left to confuse a reader.
26. As the developer, I want the contrastive encoder and its checkpoint to work exactly as before, so that brain-to-track retrieval is untouched by this effort.
27. As the developer, I want the unit tier to train on a made-up participant in a few seconds, so that the whole training path is tested without DEAP and without a real run.
28. As the developer, I want the setup docs to say that the mirror's labels are altered and where the ratings file comes from, so that a fresh setup does not repeat the mistake.
29. As the developer, I want the checkpoint download script to fetch only the contrastive checkpoint, so that setup no longer pulls classifier files nothing loads.
30. As a user of the app, I want my sessions to differ from one another, so that the session list is not 32 identical cards.
31. As a user of the app, I want a session's card label to reflect the tracks it was mostly spent on, so that the label is not decided by segment-to-segment noise.
32. As a user of the app, I want the session chart to show only the bands that exist in the data, so that I am not shown a delta band that is a filter artifact.
33. As a user of the app, I want the bands on the chart to be the ones the classifier reads, so that what I see is what the model saw.
34. As a user of the app, I want the trajectory to move track by track, so that the path is readable and not 600 jittering steps.
35. As a user of the app, I want the assistant to tell me in a few words how reliable its reading is, so that I do not take a rough read as fact.
36. As a user of the app, I want that hedge said once per analysis and not as a paragraph, so that it does not drown the answer.
37. As a user of the app, I want the assistant's explanation of a quadrant to say what the quadrant means, so that I am not told my alpha waves did something nobody measured.
38. As a user of the app, I want the assistant to narrate a session by its tracks, so that it cites "after the third track" and not a timestamp between two noisy segments.
39. As a user of the app, I want `get_model_info` to describe the model that is actually loaded, with its features, its split and its number, so that I can ask the assistant how the classifier works and get a true answer.
40. As a user of the app, I want mood playlists and relaxing-track searches to keep working after a re-seed, so that the rebuild changes the readings and not the features built on them.
41. As a future maintainer, I want the settings stated as fixed and untuned, so that I know the number was not picked from many tries on the same folds.
42. As a future maintainer, I want the roadmap to say where bigger models, Modal and a recipe search pick up, so that I know those were left out on purpose.
43. As a future maintainer, I want the glossary, the AGENTS file and the architecture guide to describe the rebuilt pipeline, so that the docs do not point at deleted code.

## Implementation Decisions

### Data and labels

- Input stays DEAP's preprocessed files from the Kaggle mirror already on disk. Only the EEG array is trusted. It was never compared with an official copy; the user accepted that.
- Valence and arousal come from DEAP's published `participant_ratings.csv`, kept beside the EEG files (gitignored, never committed: the repo is public and DEAP's licence forbids redistribution). Rows join on participant and experiment id, where a file's trial index `e` is experiment id `e + 1`.
- The files' own label array is not used for valence or arousal. Its dominance and liking columns match the ratings file in every trial, so the label loader compares them and raises if they differ: that is the check that the join is aligned.
- The DEAP loader's EEG output is unchanged, because retrieval reads it. Labels become a separate read of the ratings file.
- High is a self-rating at or above 5, the same line for every participant.

### Features

- A window is 4 seconds with no overlap, 15 a trial, 600 a participant: the same slice as a stored Segment. The 3-second pre-trial is dropped and nothing else is done with it.
- The model reads differential entropy: a Butterworth band-pass on the 4-second window, then half the log of the variance, for each of 32 channels in four bands. Theta 4 to 8, alpha 8 to 13, beta 13 to 30, gamma 30 to 45Hz. 128 features a window. No delta.
- The band-pass uses scipy's second-order-sections form. Plain numpy and scipy; no MNE or braindecode helper.
- One function in the ML kernel turns a window into its 128 features, and training and seeding both call it. One table of bands serves the model and the chart.
- A Segment's displayed band powers are that same band-passed window's variance, averaged over the 32 channels: the model reads the log of what the chart draws. The separate Welch computation is deleted. Wide bands take a bigger share of the chart than before; accepted.

### Model and training recipe

- The model is one linear layer with two outputs, the arousal and valence logits, behind a small standardization module that holds the training fold's feature mean and standard deviation as buffers. Its input is unstandardized features and its saved weights carry their own scaling.
- The training recipe is one frozen dataclass of constants in the ML kernel: band edges, window length, sample rate, label threshold, fold count, seed, learning rate, epochs, weight decay. It is part of the model, not per-machine configuration, so it is not an env var or a settings class.
- Training is full-batch AdamW on a class-balanced binary cross-entropy (a positive-class weight per axis from the training fold), a fixed number of epochs, no early stopping. CPU only; no device argument.
- Weight decay is one fixed, strong value, picked once by the implementer so that few held-out scores fall below 0.1 or above 0.9 (the evidence run saw 2 to 5% under heavy regularization, 30 to 50% without). It is not searched, and the recipe states it and the other settings as untuned.
- The folds, the metric and the loop take any torch module plus a function from a window to that module's input. Only one model exists in the repo.

### Split and metric

- One recipe, one set of models per participant, trained only on that participant's 40 trials.
- A stratified 5-fold over a participant's trials with the recipe's seed. A trial's 15 windows always sit on one side. Every trial is held out exactly once.
- One fold assignment per participant serves both axes, stratified on the quadrant of the trial's ratings. A held-out fold may lack a high or a low trial on an axis, because scoring pools a participant's held-out windows. Training raises only if a fold's training trials lack a high or a low on either axis.
- Balanced accuracy for each axis, scored per window over a participant's 600 held-out windows, then averaged over participants, with the standard deviation across participants. The mean of the two axes is the single number. Chance is 0.50 for all three. No vote over a trial.
- The expected single number is 0.57 (0.60 valence, 0.55 arousal), which is a scikit-learn baseline's figure. The first real run confirms or corrects it, and the docs record whatever it gives.
- The stratified split, the standardization and balanced accuracy are a few lines of numpy each. scikit-learn is not a direct dependency.

### Training package and command

- Training code lives in a new `cortexdj.training` package: folds, metric, the loop, and the command. It imports the ML kernel and the platform kernel and nothing else in the app. A new import-linter contract says nothing imports `cortexdj.training`.
- `train_participant(participant_id, data_dir, recipe=RECIPE)` extracts features, makes the folds, fits five models and returns them with their fold assignment and metrics as plain data. It reads no environment and no global state.
- `train-classifier` is the one way to train, declared beside `seed-sessions`. One option, `--participants`, defaulting to all 32. It loops over participants, prints the three figures with their spread, and writes the checkpoint.
- A subset run writes a checkpoint holding only those participants and overwrites the file; it does not merge into an existing one.
- The cross-subject script from the prototype branch (`refs/heads/prototype/ml-rebuild-baseline`, commit bcfe9dd) is kept in the training package as one standalone file. It is the only file that imports scikit-learn, which braindecode already installs. The training code has no leave-one-subject-out mode. Call it the cross-subject baseline script: "Baseline run" already means something else in the glossary.
- Modal is not in this effort: no wrapper, no dependency, no Volume.

### Checkpoint

- One file, about 330KB: for each participant the five fold models and which trial went to which fold, plus the metrics (per participant and the three averages) and the training recipe. It stores no predictions.
- It is made locally and never downloaded. Reading and writing the file's layout live together in the ML kernel so the two cannot drift.
- Loading compares the stored recipe with the code's, value by value, and raises naming the first setting that differs. The schema version number goes.
- Saved as state dicts and plain values so it loads with weights-only loading.

### Seeding

- `seed-sessions` loads the checkpoint and, for each segment, uses the fold model that held out that segment's trial. The stored score is the sigmoid of the logit as it comes; no calibration. The quadrant is cut at 0.5 on each score.
- It refuses to run without a checkpoint and says to run `train-classifier`. A participant the checkpoint does not hold is an error that says the same. The fallback that wrote ratings as predictions is deleted, and so is the `--model` option.
- It replaces: a participant's existing DEAP session is deleted (segments and session tracks go by cascade; tracks stay) and the fresh one inserted, so a re-seed leaves 32 sessions. The new session has a new id.
- Any thread's brain context that pointed at the replaced session has its session id cleared in the same transaction. The mood and averages on that row stay.
- The seeding function takes the data directory and the checkpoint as arguments, defaulting to the standard paths, so a test can point it at a made-up participant.

### What the app shows

- The app contract holds: 4-second segments, scores from 0 to 1, a quadrant, band powers. No migration, no API schema change, no client regeneration.
- The track is the unit the app talks in. Time in each quadrant, transitions, the dominant quadrant, centroid, dispersion, path length and the session card's label are computed over a session's 40 tracks, from the mean scores each session track already stores, not over its 600 segments. Segments stay stored and charted as they are.
- The trajectory summary keeps its shape. Its path is the 40 track points and the three-segment rolling mean goes; a transition's time is the start of the track it lands on.
- Card label cut-offs stay (90% "throughout", 60% "mostly", otherwise "mixed"), counted over tracks. Nearly every DEAP card will read "Mixed, leaning …", which is what the ratings give too.
- The one frontend change: the session chart keeps its five-band order and colours and draws only the bands present in the data. The trajectory panel is untouched.

### The assistant

- The model holder in the sessions slice now holds the loaded checkpoint (recipe and metrics). A missing file stays a startup warning that hides `get_model_info`; a recipe mismatch fails startup. `override_eeg_model` stays the test seam.
- `get_model_info` reports the loaded checkpoint: a linear layer on 128 band features, one set of models per participant, the split, the three figures and their spread, and the training recipe. Its failure message points at `train-classifier`.
- The sessions capability's instructions carry the checkpoint's single number and one wording tier from it: below 0.60 "a rough read", 0.60 to 0.75 "a fair read", 0.75 and above no hedge. One short clause, once per analysis. The number is fixed for the life of the process, so the text stays in the cached prefix (ADR 0002). With no checkpoint loaded it uses the most cautious tier.
- The Session Narrative instructions describe a path that goes track by track: transitions cite tracks, not timestamps between segments.
- The four canned quadrant explanations are rewritten to say what a quadrant means, with no claim about what the bands did. The system prompt's quadrant list loses its band claims the same way, and its band line lists the four bands with the new edges.
- The tool list's "EEGNet" wording goes.

### Deleted

- Both current classifiers (the small feature network and the CBraMod dual head with its loader), their two checkpoints on disk, the `EEG_MODEL_BACKEND` setting, the type check at inference, the schema version constant, the Welch band-power function, and the delta band.
- CBraMod stays only as the contrastive encoder's backbone. The contrastive window function, the CBraMod scale factor and `contrastive_best.pt` are untouched.

### Docs and bookkeeping

- `after-port 01` is resolved by this effort's PR, not patched on its own.
- `backend/data/DEAP_SETUP.md`: the mirror's labels are altered (drop "the same preprocessed data" and the "~24/76" split), and add the ratings-file step with its checksum.
- `README.md`, `DEVELOPMENT.md` and `AGENTS.md`: setup is `train-classifier` then `seed-sessions`; the `ml/`, model-holder and scripts paragraphs, the new training package and its import rule; no `EEG_MODEL_BACKEND`.
- The checkpoint download script fetches only `contrastive_best.pt`.
- `GLOSSARY.md`: add **Training recipe** (and keep it apart from Shared recipe); check Segment, Band power, Quadrant and Trajectory against what was built.
- `docs/ROADMAP.md`: drop the recorded 52% and any line the rebuild makes untrue; Phase 4 already carries the bake-off.
- `docs/DEMO_WORKFLOWS.md` and the architecture guide pages: fix anything that names the old classifiers or per-segment narration. The architecture guide's own skill owns larger rewrites.
- The classifier files on the GitHub Release are removed by the user, or on the user's explicit go-ahead: it is a public, outward-facing change.

## Testing Decisions

A good test here checks what a caller can see: the numbers a function returns, the rows a command leaves in the database, the text the assistant is given. It does not assert on the layer's weights, the optimizer's settings, or which helper was called. Tests fake only at declared seams (ADR 0001), never read or write `backend/data/`, and no real training run is a check: CI never touches DEAP.

Three seams, all existing in kind:

1. **The DEAP files, as a directory argument.** The existing seam for retrieval, now also for training and seeding. A shared builder writes one made-up participant in DEAP's shape (a small pickle plus a ratings file) under `tmp_path`, with a signal planted so that a linear model can find it. Prior art: the mock DEAP file in the DEAP loader's unit test and the retrieval service's `deap_dir=` tests.
2. **`override_eeg_model`.** Already declared. It now takes a loaded checkpoint or `None`. Prior art: the model-holder and sessions-capability unit tests.
3. **The database, through the existing fixtures.** `make_deps()` in the unit tier, the `client` fixture in the integration tier. Prior art: the sessions tools and API integration tests.

What is tested through them:

- **Training, unit tier, through `train_participant` on the made-up participant** (a few seconds): no trial straddles a fold; every trial is held out exactly once; the planted signal scores well above chance and the same data with shuffled labels scores near 0.50; a rerun gives the same metrics; a fold whose training trials are missing a class on an axis raises.
- **Checkpoint, unit tier:** save then load gives the same predictions; a changed recipe raises at load and names the setting; a missing file is a warning and an empty holder, a mismatch fails the startup hook.
- **Features, unit tier:** a window gives 128 features and four displayed bands with no delta; a sine in one band lands in that band; the displayed power and the feature agree (the feature is half the log of the variance).
- **Labels, unit tier:** ratings join on participant and experiment id; a ratings file whose dominance and liking disagree with the EEG file raises; the EEG the loader returns is unchanged.
- **Seeding, integration tier, through the seed function** with the made-up participant and a checkpoint trained on it in the test: each segment's score equals the held-out fold model's output; seeding twice leaves one session for that participant; a thread's brain context that pointed at the replaced session loses its session id; no checkpoint, or a participant the checkpoint lacks, fails with the `train-classifier` message and writes nothing.
- **Trajectory, unit tier,** in the existing trajectory tests: the summary is computed over tracks; segment noise inside a track makes no transition.
- **Sessions capability and tools,** in the existing tests: the hedge tier follows the checkpoint's number at each cut-off; `get_model_info` is hidden with no checkpoint and reports the recipe and metrics with one.
- **Session card label, integration tier,** in the existing sessions API tests: counted over tracks.
- **Import rule:** the new contract runs with the existing import-linter hook.
- **Contrastive path:** its existing tests pass unchanged; that is the check that retrieval was not touched.
- **Frontend:** lint and typecheck. The chart change has no test tier in this repo; it is checked by eye after a re-seed.

Tests that go with the code they cover: the unit tests for the two deleted classifiers, and delta in the factories and the preprocessing tests.

The eval tier is not part of this spec's checks. The session-narration and explanation cases may shift once the instructions change; run it only if the user asks.

Checked by the user after merge, not by a test: the first real `train-classifier` run's three figures, the spread of real scores at the chosen weight decay, how the trajectory panel looks with a 40-point path over 600 dots, and whether the assistant keeps the hedge to one clause.

## Out of Scope

- Retraining the contrastive EEG↔CLAP encoder (blocked on `audio-decoding 01`).
- Bigger models, raw-window architectures, a fine-tuned CBraMod as the classifier, and the bake-off between them.
- Modal: the wrapper, the dependency, the Volume.
- An autoresearch loop or any search over the training recipe. The recipe argument is the opening; nothing uses it yet.
- Calibrating the scores, a vote over a trial's windows, per-participant label thresholds.
- A leave-one-subject-out mode in the training code.
- DEAP's raw recordings, other datasets, and the user's own Muse recordings.
- Verifying the EEG against an official DEAP copy.
- Database migrations, API schema changes, and any frontend change beyond the chart drawing only the bands present.
- The other open tickets (`after-port 02` to `04`, `audio-decoding 01`).

## Further Notes

**Calls this spec made that no ticket settled.** Each is cheap to change; say so if one is wrong.

- *One fold assignment, stratified on the quadrant.* The tickets say "stratified 5-fold" and "one model with two outputs", but the baseline and evidence scripts split each axis separately, which a two-output model cannot do. Stratifying on the quadrant is the natural single split (confirmed by the user). With the most lopsided participant at 6 trials in the smaller arousal class, a held-out fold can end up without one class on an axis; that is allowed, because the score pools a participant's 600 held-out windows. Training raises only when a fold's training trials lack a class, which cannot happen on DEAP under this split: each quadrant is spread across the folds, so no fold holds all of a class.
- *No oversampling.* Class imbalance is handled by the class-balanced loss and by reporting balanced accuracy. For a full-batch linear model, repeating the rare class's windows is the same as weighting them in the loss.
- *Balanced accuracy over a participant's 600 held-out windows pooled,* not per fold then averaged. This is what the evidence scripts did and what gave 0.60 and 0.55.
- *The dominance and liking comparison is a hard check at load,* not just a note in the setup doc.
- *A subset training run overwrites the checkpoint,* and seeding a participant it lacks is an error.
- *Brain context:* the ticket's default is taken (the seed clears the replaced session's id).
- *No checkpoint loaded means the most cautious hedge tier.*
- *"Training recipe" goes in the glossary* as its own term.

**Unverified, carried from the tickets.** No torch model has been run: 0.57, the one-minute run time and the 330KB file size are all estimates. That the standardization buffers survive weights-only loading is assumed and is the first thing the checkpoint test proves. The hedge cut-offs (0.60, 0.75) are picked, not sourced.

**Limits to state wherever the number is quoted.** Within-subject on DEAP is within one sitting; a decoder trained one day and used another is a harder problem this number does not cover. Comparing many recipes on the same folds would drift the number upward; few knobs is the guard.

**Research the tickets cite** lives on unmerged branches: `refs/heads/research/ml-rebuild-braindecode-mne-deap`, `refs/heads/research/ml-rebuild-deap-benchmarks`, `refs/heads/research/ml-rebuild-modal-fan-out`, `refs/heads/research/ml-rebuild-signal-libraries`. The score-spread evidence is in `evidence/` here.

**Size.** This is one effort but more than one PR's worth of review. A natural cut for `to-tickets`: labels and features; training package, checkpoint and command; seeding and the model holder; track-level analytics and the assistant's wording; the chart; docs and deletions.
