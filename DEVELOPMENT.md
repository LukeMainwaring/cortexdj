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
uv run --directory backend pytest              # run all tests
uv run --directory backend pytest -v           # verbose output
uv run --directory backend pytest tests/test_preprocessing.py  # single file
```

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

# Training runs
modal run backend/scripts/modal_train.py                                      # full training on A10G
modal run backend/scripts/modal_train.py --args="--quick"                     # quick test run
modal run backend/scripts/modal_train.py --args="--model eegnet"              # train EEGNet instead
modal run backend/scripts/modal_train.py --gpu a100                           # faster GPU
modal run backend/scripts/modal_train.py --command compare-models             # compare both
```

DEAP source files (~3.1 GB of `.dat`) live in a persistent `cortexdj-deap` Modal Volume seeded once via the loop above. Subsequent training runs attach the volume instantly instead of re-uploading. The first GPU run regenerates `data/deap/.cache/*.npz` (preprocessing cache) inside the volume; `modal_train.py` calls `deap_volume.commit()` after training so that cache persists for later runs. Checkpoints are automatically downloaded to `backend/data/checkpoints/` when the run completes.

### Database Seeding

```bash
uv run --directory backend seed-sessions                          # seed all 32 DEAP participants
uv run --directory backend seed-sessions --participants 1 2 3     # seed specific participants
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
