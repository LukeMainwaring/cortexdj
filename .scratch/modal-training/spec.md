# Spec: Train the quadrant classifier on Modal

**Status:** resolved (2026-11-22, PR #100)

Settled in a Grill with Docs session on 2026-11-22 (fourteen decisions, no glossary entry, no ADR). It builds on two decision tickets of the ml-rebuild map: `ml-rebuild 03` (Modal's current setup for small fan-out runs) and `ml-rebuild 08` (where training code lives, and how Modal fits). The reference for Modal's API is `docs/research/modal-fan-out-training.md` on `refs/heads/research/ml-rebuild-modal-fan-out`, read from Modal's docs at SDK 1.6.4.

Small enough to implement directly: no implementation tickets.

## Problem Statement

Training the quadrant classifier runs only on the developer's laptop. That is fine today: one linear layer per fold, all 32 participants in about a minute. It stops being fine with the first model that is slow to train, and the next effort (comparing the linear layer with an MLP, a real EEGNet and a fine-tuned CBraMod) starts with exactly those.

Modal is where that work is meant to run, and nothing about Modal has been proven for this repo. The earlier wrapper was deleted with the old training code, and it took about 15 minutes to start each run. The replacement design exists only on paper: the image build from the lockfile, Python 3.14 on Modal, the cold start, the cost, and whether a participant's models come back intact are all unmeasured. Finding those out at the same time as adding new models would mix two sources of failure.

## Solution

A second way to make the same checkpoint. `train-classifier` is unchanged and stays the documented setup step. Beside it, a new command, `train-classifier-modal`, trains each participant in its own Modal container at the same time, collects the results on the laptop, and writes the same `quadrant_classifier.pt` to the same place.

The two commands take the same `--participants` option and print the same three lines: balanced accuracy for arousal, for valence, and their mean, each with its spread across participants. The Modal command is proven correct when those lines match a local run's to three decimals.

DEAP reaches Modal once, by hand: the developer uploads the 32 EEG files and the ratings file to a private Modal Volume, which every container mounts read-only.

Nothing else changes. No new model, no recipe search, no GPU, and the app, `seed-sessions` and the checkpoint's contents are untouched.

## User Stories

1. As the developer, I want a command that trains every participant's quadrant classifiers on Modal, so that training no longer depends on what my laptop can do.
2. As the developer, I want `train-classifier` left exactly as it is, so that setup still needs no cloud account and takes about a minute.
3. As the developer, I want the Modal command to take the same `--participants` option as `train-classifier`, so that I do not have to remember two ways to name a subset.
4. As the developer, I want the Modal command to print the same three lines as `train-classifier`, so that I can compare the two runs by eye.
5. As the developer, I want the Modal run to write the same checkpoint file to the same path, so that `seed-sessions` and the app need no change to use it.
6. As the developer, I want a Modal run's three figures to match a local run's to three decimals, so that I know the wrapper changed where training runs and nothing else.
7. As the developer, I want a third-decimal difference explained before the work merges, so that a real divergence between the two machines is not waved through.
8. As the developer, I want each participant trained in its own container at the same time, so that a full run takes about as long as one participant.
9. As the developer, I want the training function itself to have no Modal import, so that the code that trains locally is the same code that trains on Modal.
10. As the developer, I want the Modal wrapper to be the only file that knows about Modal, so that removing or replacing Modal later touches one file.
11. As the developer, I want the wrapper inside the training package, so that mypy and ruff check it like the rest of the training code.
12. As the developer, I want the rule that nothing in the app imports the training package to hold unchanged, so that adding Modal does not loosen the import contracts.
13. As the developer, I want `modal` to be a development dependency only, so that the app's runtime install does not grow.
14. As the developer, I want the Modal image built from the backend lockfile, so that Modal trains with exactly the library versions my laptop does.
15. As the developer, I want a code edit not to rebuild the image's dependency layer, so that a rerun after a small change starts quickly.
16. As the developer, I want to see the image build and each container's progress while the command runs, so that a slow or stuck run is visible.
17. As the developer, I want a participant that fails to be retried, so that one preempted container does not cost me the run.
18. As the developer, I want the whole run to fail, with no checkpoint written, when a participant still fails after its retries, so that I never get a checkpoint missing a participant.
19. As the developer, I want a failed Modal run to leave my existing checkpoint alone, so that a broken run cannot take away a working setup.
20. As the developer, I want DEAP uploaded to Modal once, by one documented command I run myself, so that I control when licensed data leaves my machine.
21. As the developer, I want the DEAP Volume mounted read-only, so that a training run cannot alter the data every later run reads.
22. As the developer, I want the Modal run to fail fast with a clear message when the Volume does not exist, so that I know to do the upload.
23. As the developer, I want the setup docs to say what DEAP's licence says and does not say about a cloud copy, so that the reasoning behind the upload is on record.
24. As the developer, I want the docs to say the Volume must be deleted if access to DEAP is ever withdrawn, so that the one obligation the licence does state is not forgotten.
25. As the developer, I want the commands I run to verify the work listed in order, so that verification does not depend on remembering this conversation.
26. As the developer, I want the measured cold start, run time and cost recorded in the docs, so that the next effort plans against real figures.
27. As the developer, I want the Modal research brought into the repo and corrected by what the first run shows, so that its unverified list stops being a list of guesses.
28. As the developer, I want the roadmap's 15-minute cold-start note replaced with the measured figure, so that the roadmap describes the current wrapper.
29. As the developer, I want CI to stay free of Modal, so that a pull request never needs a cloud account or spends money.
30. As the developer of the next effort, I want a working fan-out over participants already in place, so that comparing models is a matter of adding models.

## Implementation Decisions

**Scope**

- Modal is a second way to produce the quadrant-classifier checkpoint, not the default. It becomes the default only when a model arrives that is too slow to train locally.
- `train-classifier`, `train_participant`, the training recipe, the checkpoint's contents and its loader do not change.

**The wrapper**

- One new module in the training package holds everything Modal: the app, the image, the Volume reference, the remote function, and the command's entry point. No other module imports `modal`.
- `modal` goes in the `dev` dependency group. The image is built without the dev group, which works because Modal supplies its own client inside containers.
- No import-linter contract changes. The existing contracts restrict imports between `cortexdj` modules, and the new module sits inside the training package and imports only the ML kernel, the platform kernel and its own package.
- The remote function takes a participant id and the recipe's default, calls `train_participant` with the Volume's mount path as the data directory, and returns the resulting `ParticipantModels` unchanged. No conversion to plain dicts or bytes: both sides run this repo's code from the same lockfile.
- The wrapper imports the training code at module level like any other module. The training code itself has no Modal import.
- Both commands read `--participants` through one shared parser function in the local command's module, so the option cannot drift between them.

**The command**

- A second console script, `train-classifier-modal`, declared beside `train-classifier`. It parses `--participants` with the same argparse definition as the local command (same default of all 32) and starts the Modal app from Python with `app.run()`, inside `modal.enable_output()` so the image build and progress are shown.
- This form was chosen over `modal run` because a `modal run` entrypoint accepts only single values and could not take `--participants 1 2 3`.
- It maps the remote function over the chosen participant ids, collects the results keyed by participant id, builds the checkpoint with the same constructor the local command uses, saves it to the same checkpoint path with the same save function, and prints the same summary text followed by the path written.
- Results are collected in full before anything is written. A participant that fails after its retries raises, the command exits with the error, and no file is written, so an existing checkpoint is left as it was.
- The entry point is a function, not module-level code, so the module can be imported inside a container without starting a run.
- Modal's docs show `map` only under `modal run`, not inside `app.run()`. It is expected to behave the same; the first run proves it.

**Image and machine**

- The image is Debian slim on Python 3.14 with the whole backend lockfile synced without the dev group, plus the `cortexdj` package added as local Python source at container start. A separate training-only dependency group was turned down: a smaller image, but a second list to keep in sync.
- The lockfile directory is found from the wrapper module's own location, not from the current directory, because the command is run with `uv run --directory backend`.
- Each participant runs on CPU with 1 core, 2GiB of memory, a 10-minute timeout and 2 retries. These are estimates from a local participant taking about 2 seconds and holding about 100MB; the first run corrects them.
- No GPU option. It arrives with the first model that needs one.

**Data**

- A private Modal Volume named `cortexdj-deap`, created with the default Volume version, holds the 32 EEG files and `participant_ratings.csv` in one directory, laid out as `train_participant` expects a data directory.
- The developer creates and fills it by hand with Modal's CLI. There is no upload script. An old Volume of the same name from the deleted wrapper is deleted first, since it predates the ratings file.
- The remote function mounts it read-only and does not create it when missing, so a missing Volume fails at start.

**Licence**

- DEAP's EULA (read from the Internet Archive's June 2025 copy; the official site is down) does not mention cloud storage. It forbids distributing the dataset, sharing download credentials and commercial use, and it requires deleting any copies if access is withdrawn.
- The developer's decision is to proceed: a Volume only their account can read distributes nothing. The DEAP setup doc records this reading, the strict reading it does not rule out, and the deletion duty.
- The EULA's academic-use and signed-licensee terms apply to the local copy as much as to the Volume. This spec does not change that position and the setup doc states it.

**Docs**

- `DEVELOPMENT.md`: a short Modal section with the one-time setup (log in, create the Volume, upload), the run command, and the measured cold start, run time and cost.
- `backend/data/DEAP_SETUP.md`: the licence paragraph above.
- `docs/research/modal-fan-out-training.md`: brought onto `main` from the research branch, indexed in the research README. Research notes are dated snapshots and are not maintained, so the body is left as read that day, with one note on top saying what was built and the two ways it differs from the sketches: the fan-out is over participants, not leave-one-subject-out folds, and the command is a console script, not `modal run`. What the first run measures goes in `DEVELOPMENT.md`, not here.
- `AGENTS.md`: one sentence in the training paragraph naming the wrapper and its command.
- `docs/ROADMAP.md`: the "~15-min cold-start" note replaced by the measured figure, and the bake-off line updated to say the wrapper exists.

**Merge**

- The pull request stays a draft until the developer has run the verification below, the three lines match, and the measured figures are in the docs.

## Testing Decisions

A good test checks what a caller can see. Here that is the three lines the command prints and the checkpoint file it writes, and both exist only after a real Modal run. So the check for this work is the developer's own run, not an automated test.

- **No automated test touches Modal, and no test mocks it.** The wrapper's own logic is one call to `train_participant` and one loop over constructors and a save function that the existing unit tests already cover. A test with Modal mocked out would assert on the mock.
- **No new seam.** The data directory argument of `train_participant` is the existing seam, and the wrapper uses it as it stands by passing the Volume's mount path.
- **Static checks cover the file:** mypy strict and ruff through the existing commands, and the import-linter hook confirms the contracts still hold with the new module present.
- **Existing tiers pass unchanged:** the unit tier, the integration tier and the pre-commit hooks. That is the check that the local training path was not touched. Prior art for what is already covered: the training unit tests that run `train_participant` on a made-up participant, and the checkpoint unit tests for save then load.
- **CI is unchanged** apart from installing `modal` with the dev group.

Verified by the developer, in this order, before merge:

1. Log in to Modal if the stored token has expired.
2. Delete any old `cortexdj-deap` Volume, create a new one, and upload the DEAP directory.
3. Run `train-classifier` and note its three lines.
4. Run `train-classifier-modal` and compare its three lines with step 3: arousal, valence and mean, and their spreads, equal to three decimals. The reference from the first full local run is arousal 0.558 ± 0.053, valence 0.609 ± 0.080, mean 0.583 ± 0.053.
5. Run `train-classifier-modal --participants 1 2 3` and check it writes a checkpoint holding only those three.
6. Record the image build time on the first run, the cold start and run time on a second run, and the cost Modal reports.

A third-decimal difference in step 4 is a finding to explain, not a pass. The two machines differ in processor and operating system, so the last digits of the floats may differ; three decimals of a balanced accuracy over thousands of windows should not.

## Out of Scope

- New models (an MLP, a real EEGNet, a fine-tuned CBraMod) and the comparison between them.
- A recipe search or an autoresearch loop, and the nested or held-out split it needs to stay honest.
- A GPU option, a detached run that survives the laptop closing, and results returned through a Volume.
- Making Modal the default training path.
- A training-only dependency group or any other work to shrink the image.
- An upload script or any automation of the Volume.
- Training the contrastive encoder on Modal.
- Asking DEAP's administrators about cloud storage. The developer chose to proceed on the reading above.

## Further Notes

Unverified until the developer's run:

- That the lockfile syncs on Modal under Python 3.14, and how long the image takes to build. The lockfile resolves the Linux CUDA wheels, so the image may be large for a CPU job. If the build time is a problem, a CPU-only torch index for the image is the first thing to try, as a follow-up.
- That `ParticipantModels`, which holds torch modules, comes back through Modal's serialization intact. If it does not, the fallback is to return each model's state dict and rebuild the modules on the laptop, which is the conversion the checkpoint's save and load already do.
- That `map` inside `app.run()` behaves as it does under `modal run`.
- That three decimals agree between macOS on ARM and Linux on x86.
- The developer's Modal account: whether the stored token is still valid, the plan's container limit against 32 at once, and whether an old Volume exists.

The licence position is the developer's call, recorded here so it is not re-argued: the EULA is silent on cloud storage, and only the dataset's administrators could settle the strict reading.
