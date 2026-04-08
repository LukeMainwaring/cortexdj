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

### Synthetic Data

```bash
uv run --directory backend generate-synthetic                    # 32 participants, 40 trials each
uv run --directory backend generate-synthetic --participants 8   # fewer participants for quick testing
```

### Model Training

```bash
uv run --directory backend train-model
uv run --directory backend train-model --epochs 50 --lr 0.001 --folds 5
```

Model checkpoints saved to `backend/data/checkpoints/` (gitignored).

### Database Seeding

```bash
uv run --directory backend seed-sessions                          # seed all 32 participants
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
