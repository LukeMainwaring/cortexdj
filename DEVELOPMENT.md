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

Two model backends are available:

```bash
# CBraMod pretrained with LOSO CV (default — 50 epochs, all 32 folds)
uv run --directory backend train-model

# Quick dev run (10 epochs, 3 folds)
uv run --directory backend train-model --quick

# EEGNet instead
uv run --directory backend train-model --model eegnet

# Custom configuration
uv run --directory backend train-model --epochs 100 --batch-size 128 --max-folds 5

# Compare both models (loads checkpoints by default, --retrain to train fresh)
uv run --directory backend compare-models
```

Model checkpoints saved to `backend/data/checkpoints/` (gitignored). CBraMod is the default runtime backend. Set `EEG_MODEL_BACKEND=eegnet` in `.env` to use the lightweight model instead.

**Label binarization (`--label-split`).** DEAP's 1–9 Likert self-reports are skewed when thresholded at a fixed value. Strategies:

- `median_per_subject` (default, recommended): each subject is split at their own Likert median on each axis, giving balanced labels per fold and removing per-subject rating-scale bias.
- `median_global`: pooled median across all 32 subjects. Slightly less balanced per-fold but deterministic across subjects.
- `fixed_5`: `>= 5` threshold. Produces a ~24/76 high/low split on DEAP — only useful for reproducing papers that adopted this convention.

Strategies are cached independently (`_CACHE_VERSION` encodes the strategy in the `.npz` filename), so switching is free after the first build.

**Reading the output.** Each epoch logs `Val acc A/V`, `macro-F1`, and per-class `pred A [low, high] V [low, high]` counts. The fold summary prints both accuracy and macro-F1 columns plus a `Mean recall` line. The headline metric is `Avg F1` — raw accuracy on balanced labels can hide majority-class predictions. `compare-models` always renders a `MajorityBaseline` reference row from the dataset labels; a trained model should beat it on `Avg F1` by at least +0.05. If a fold produces non-finite loss, the loop skips the batch and logs `[Phase N] Epoch K/T: skipped M non-finite-loss batches` — zero such warnings is the expected state.

**Local MPS training.** EEGNet quick runs work well on Apple Silicon (`--quick` finishes in under a minute per fold). `non_blocking=True` data transfers and `set_float32_matmul_precision("high")` are gated on CUDA only — PyTorch 2.9–2.11 has had intermittent MPS async-correctness regressions on the pinned-memory path, and TF32 is a CUDA-only matmul setting. Full 32-fold CBraMod training still wants a CUDA GPU; see the Modal section below.

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
caffeinate -dim modal run backend/scripts/modal_train.py                       # full training on A10G
modal run backend/scripts/modal_train.py --args="--quick"                     # quick test run
modal run backend/scripts/modal_train.py --args="--model eegnet"              # train EEGNet instead
modal run backend/scripts/modal_train.py --gpu a100                           # faster GPU
modal run backend/scripts/modal_train.py --command compare-models             # compare both
```

**Preemption resume.** Fold-level progress is persisted under `backend/data/deap/.train_state/` (which rides along on the `cortexdj-deap` Modal volume), so a preempted run restarts at the last completed fold rather than starting over. A fresh run with matching hyperparameters auto-resumes; pass `--args="--no-resume"` to wipe prior state and start clean.

DEAP source files (~3.1 GB of `.dat`) live in a persistent `cortexdj-deap` Modal Volume seeded once via the loop above. Subsequent training runs attach the volume instantly instead of re-uploading. The first GPU run regenerates `data/deap/.cache/*.npz` (preprocessing cache) inside the volume; `modal_train.py` calls `deap_volume.commit()` after training so that cache persists for later runs. Checkpoints are automatically downloaded to `backend/data/checkpoints/` when the run completes.

CUDA runs auto-configure themselves: batch size defaults to 128 (vs. 64 on MPS/CPU), bf16 mixed precision is enabled, `cudnn.benchmark` is on, and DataLoader uses 8 workers with `prefetch_factor=4`. No extra flags are needed for the common case — `modal run backend/scripts/modal_train.py` is the happy path. Override with `--args="--batch-size 192"` etc. if you want to push the A10G harder.

### Database Seeding

```bash
uv run --directory backend seed-sessions                          # seed all 32 DEAP participants (cbramod by default)
uv run --directory backend seed-sessions --participants 1 2 3     # seed specific participants
uv run --directory backend seed-sessions --model eegnet           # use lightweight EEGNet checkpoint instead
```

## Common Commands

### Backend

```bash
uv sync --directory backend                   # Install dependencies
docker compose up -d                          # Start PostgreSQL + backend
uv run --directory backend pre-commit run --all-files  # Linting + type checking
```

### Frontend

```bash
pnpm -C frontend install
pnpm -C frontend dev                          # Start dev server on port 3003
pnpm -C frontend lint                         # Lint with ultracite
pnpm -C frontend format                       # Format with ultracite
pnpm -C frontend generate-client              # Regenerate API client
```
