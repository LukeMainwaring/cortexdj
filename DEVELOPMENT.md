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
cd backend && uv sync
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

## Pre-commit hooks

```bash
uv run pre-commit run --all-files
```

## Database migrations

```bash
# Create a new migration
cd backend && ./scripts/create-db-revision-docker.sh "<migration_message>"

# Apply all pending migrations
cd backend && ./scripts/migrate-docker.sh

# Roll back one migration
cd backend && ./scripts/downgrade-db-revision-docker.sh
```

## API Client Generation

After modifying backend API endpoints:

```bash
# Ensure backend is running
docker compose up -d

# Regenerate client
cd frontend && pnpm generate-client
```

## ML Development

### Synthetic Data

```bash
uv run generate-synthetic                    # 32 participants, 40 trials each
uv run generate-synthetic --participants 8   # fewer participants for quick testing
```

### Model Training

```bash
uv run train-model
uv run train-model --epochs 50 --lr 0.001 --folds 5
```

Model checkpoints saved to `backend/data/checkpoints/` (gitignored).

### Database Seeding

```bash
uv run seed-sessions                          # seed all 32 participants
uv run seed-sessions --participants 1 2 3     # seed specific participants
```

## Common Commands

### Backend

```bash
cd backend && uv sync                         # Install dependencies
docker compose up -d                          # Start PostgreSQL + backend
uv run pre-commit run --all-files            # Linting + type checking
```

### Frontend

```bash
cd frontend && pnpm install
cd frontend && pnpm dev                       # Start dev server on port 3003
cd frontend && pnpm lint                      # Lint with ultracite
cd frontend && pnpm format                    # Format with ultracite
cd frontend && pnpm generate-client           # Regenerate API client
```
