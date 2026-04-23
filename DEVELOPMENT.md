# Development

## Tools

This project uses:

- **[uv]** - Fast Python package installer and resolver
- **[Docker]** - Container platform for local development
- **[Ruff]** - Fast Python linter and formatter
- **[mypy]** - Static type checker
- **[pre-commit]** - Git hook framework

[uv]: https://docs.astral.sh/uv/
[Docker]: https://docs.docker.com/get-docker/
[Ruff]: https://github.com/astral-sh/ruff
[mypy]: https://mypy-lang.org/
[pre-commit]: https://pre-commit.com/

## Setup

Install dependencies:

```bash
uv sync --directory backend
```

Install pre-commit hooks:

```bash
uv run --directory backend pre-commit install
```

## Pre-commit hooks

```bash
uv run --directory backend pre-commit run --all-files
```

## Tests

```bash
uv run --directory backend pytest              # unit tests (eval suite excluded)
uv run --directory backend pytest -v           # verbose output
uv run --directory backend pytest tests/test_preprocessing.py  # single file
uv run --directory backend pytest -m eval      # real-model brain_agent eval suite (opt-in)
```

The `eval` marker gates tests that call the real OpenAI API via `brain_agent` — the default `pytest` invocation excludes them via `addopts = "-m 'not eval'"`. Use them as a nightly safety net on `main` or manual spot-checks, not on every PR. See `.claude/rules/backend/pydantic-ai.md` and `backend/tests/evals/` for the suite layout.

## Database migrations

```bash
# Create a new migration
./backend/scripts/create-db-revision-docker.sh "<migration_message>"

# Apply all pending migrations
./backend/scripts/migrate-docker.sh

# Roll back one migration
./backend/scripts/downgrade-db-revision-docker.sh
```

## API Client Generation

After modifying backend API endpoints:

```bash
# Ensure backend is running
docker compose up -d

# Regenerate client
pnpm -C frontend generate-client
```

## ML Development

### DEAP Dataset

See [backend/data/DEAP_SETUP.md](backend/data/DEAP_SETUP.md) for download instructions. Place `.dat` files in `backend/data/deap/`.

### Model Training

Three trainable models live side by side: the quadrant classifier (EEGNet or CBraMod backbone) and the contrastive EEG↔CLAP encoder.

```bash
# -- Classifier (arousal/valence quadrant prediction) --

# CBraMod pretrained with LOSO CV (default — 50 epochs, all 32 folds)
uv run --directory backend train-model

# Quick dev run (10 epochs, 3 folds)
uv run --directory backend train-model --quick

# EEGNet instead
uv run --directory backend train-model --model eegnet

# Compare both models (loads checkpoints by default, --retrain to train fresh)
uv run --directory backend compare-models

# -- Contrastive encoder (EEG ↔ CLAP audio retrieval) --

# Prereqs: hand-curated deap_stimuli.json is committed; run fetch-deap-audio
# once to resolve each DEAP stimulus to a cached iTunes m4a preview.
uv run --directory backend fetch-deap-audio

# Full training: 30 epochs, 24/4/4 subject split, SequentialLR warmup+cosine,
# SimCLR-style projection head, TensorBoard + embedding projector snapshot
uv run --directory backend train-contrastive

# Quick smoke (5 epochs × 3 train / 1 val / 1 test subjects) — ~45s on MPS
uv run --directory backend train-contrastive --quick

# Gradient accumulation for larger effective batches on MPS
uv run --directory backend train-contrastive --grad-accum 4

# Inspect the run
uv run --directory backend tensorboard --logdir backend/data/tensorboard_runs
```

Model checkpoints saved to `backend/data/checkpoints/` (gitignored). CBraMod is the default runtime backend for the quadrant classifier. Set `EEG_MODEL_BACKEND=eegnet` in `.env` to use the lightweight model instead. The contrastive checkpoint (`contrastive_best.pt`) is loaded lazily on the first `/api/sessions/{id}/similar-tracks` request or the first agent call to `retrieve_tracks_from_brain_state`.

**Label binarization (`--label-split`).** DEAP's 1–9 Likert self-reports are skewed when thresholded at a fixed value. Strategies:

- `median_per_subject` (default, recommended): each subject is split at their own Likert median on each axis, giving balanced labels per fold and removing per-subject rating-scale bias.
- `median_global`: pooled median across all 32 subjects. Slightly less balanced per-fold but deterministic across subjects.
- `fixed_5`: `>= 5` threshold. Produces a ~24/76 high/low split on DEAP — only useful for reproducing papers that adopted this convention.

Strategies are cached independently (`_CACHE_VERSION` encodes the strategy in the `.npz` filename), so switching is free after the first build.

### GPU Training (Modal)

Full LOSO with CBraMod takes 12+ hours on Apple Silicon. Use [Modal](https://modal.com) for a one-off GPU run (~1 hour on A10G, ~$1-2):

```bash
# One-time setup — auth + DEAP Volume seed
# (modal is a regular backend dep, installed via `uv sync --directory backend`)
modal setup
modal volume create cortexdj-deap

# Upload one .dat per invocation so each gets a fresh S3 token (bulk uploads
# blow past the ~1hr presigned URL TTL on slow uplinks). Idempotent: rerun
# the loop after a drop and it skips files already in the volume. We exclude
# .cache/ — it's regenerable preprocessing output and the GPU container will
# rebuild it on first run, then commit it back to the volume.
caffeinate -dim bash -c 'for f in backend/data/deap/s*.dat; do
    echo "Uploading $(basename "$f")..."
    modal volume put cortexdj-deap "$f" "/$(basename "$f")"
done'
modal volume ls cortexdj-deap /   # sanity check — should list 32 .dat files

# Training runs — wrap long runs in caffeinate so macOS can't sleep
# and drop the client connection mid-run
caffeinate -dim modal run backend/scripts/modal_train.py                       # classifier, full LOSO on A10G
modal run backend/scripts/modal_train.py --args="--quick"                     # quick test run
modal run backend/scripts/modal_train.py --args="--model eegnet"              # train EEGNet instead
modal run backend/scripts/modal_train.py --gpu a100                           # faster GPU
modal run backend/scripts/modal_train.py --command compare-models             # compare both

# Contrastive encoder on Modal (ships deap_stimuli.json + audio_cache/ with the image)
caffeinate -dim modal run backend/scripts/modal_train.py --command train-contrastive
modal run backend/scripts/modal_train.py --command train-contrastive --args="--quick"
```

**Preemption resume.** Fold-level progress is persisted under `backend/data/deap/.train_state/` (which rides along on the `cortexdj-deap` Modal volume), so a preempted run restarts at the last completed fold rather than starting over. A fresh run with matching hyperparameters auto-resumes; pass `--args="--no-resume"` to wipe prior state and start clean.

DEAP source files (~3.1 GB of `.dat`) live in a persistent `cortexdj-deap` Modal Volume seeded once via the loop above. Subsequent training runs attach the volume instantly instead of re-uploading. The first GPU run regenerates `data/deap/.cache/*.npz` (preprocessing cache) inside the volume; `modal_train.py` calls `deap_volume.commit()` after training so that cache persists for later runs. Checkpoints are automatically downloaded to `backend/data/checkpoints/` when the run completes.

### Autoresearch (autonomous EEGNet iteration)

A [Karpathy-style](https://github.com/karpathy/autoresearch) overnight experiment loop for the quadrant classifier — an AI coding agent modifies a single training file, runs a 15-minute experiment on Modal, reads the resulting `avg_macro_f1`, and decides keep-or-revert. Targets the EEGNet pipeline on a fixed 28/4 DEAP subject split.

```bash
# Run one experiment manually (~30 min end-to-end: ~15 min cold-start + 15 min training)
uv run --directory backend python scripts/run_autoresearch.py
# or equivalently:
modal run backend/scripts/modal_autoresearch.py

# Local dry-run on CPU — no Modal, short budget — for sanity-checking train.py edits
WALL_CLOCK_BUDGET_SECONDS=30 AUTORESEARCH_RUN_DIR=/tmp/ar_local \
  uv run --directory backend python -m autoresearch.train
```

To run the agent loop, point any coding agent (Claude Code, Codex) at `backend/autoresearch/program.md` in a fresh session. The agent reads that file, forms a hypothesis, edits `train.py`, invokes the wrapper, and tails `experiments/experiments.jsonl` to decide keep-or-revert. Overnight (8h) budgets roughly 12–16 experiments.

Results land in `backend/autoresearch/experiments/` (gitignored): one JSONL row per run, a `best.json` pointer to the current champion, and per-run artifacts (`train.py` snapshot, `stdout.log`, `metrics.json`). See `backend/autoresearch/README.md` for the full workflow, the four contracts the agent must preserve, and design rationale.

### Database Seeding

```bash
# Seed the `sessions` and `eeg_segments` tables from DEAP .dat files
uv run --directory backend seed-sessions                          # seed all 32 DEAP participants (cbramod by default)
uv run --directory backend seed-sessions --participants 1 2 3     # seed specific participants
uv run --directory backend seed-sessions --model eegnet           # use lightweight EEGNet checkpoint instead

# Seed the `track_audio_embeddings` pgvector index for EEG↔CLAP retrieval.
# Pool = (user Spotify saved tracks) ∪ (12 genre-seed searches). Resolves each
# candidate to an iTunes 30s m4a via services/audio_catalog.resolve_preview,
# embeds with LAION-CLAP, upserts via pgvector. Misses go to
# backend/data/track_index_miss_log.jsonl — no retry loops.
# Requires Spotify user OAuth (connect via Settings in the UI first).
uv run --directory backend seed-track-index                       # default limit from TRACK_INDEX_POOL_SIZE
uv run --directory backend seed-track-index --limit 500           # smaller pool for dev
uv run --directory backend seed-track-index --skip-library        # genre seeds only (no user OAuth)
```

One-off verification scripts live under `backend/scripts/` with usage in their own docstrings (e.g. `probe_audio_catalog.py` for tuning the iTunes matcher against a live Spotify library).
