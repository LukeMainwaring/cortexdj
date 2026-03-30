# Backend Patterns

Python/FastAPI conventions for the CortexDJ backend.

## Code Style

- Use lowercase with underscores for filenames (e.g., `brain_agent.py`, `eeg_processing.py`)
- Use modern Python syntax: `| None` over `Optional`, `list` over `List`
- Use f-strings for logging: `logger.info(f"Created {item.id}")`
- Use descriptive variable names with auxiliary verbs (e.g., `is_active`, `has_permission`)
- Type hints required on all functions

## Architecture

- Use `def` for pure functions, `async def` for I/O operations
- Use FastAPI's dependency injection for shared resources (db sessions, ML models)
- All database operations are async using `AsyncSession`
- Keep route handlers thin: push business logic to `services/`, DB logic to `models/`
- Import service modules with a named alias in routers: `from cortexdj.services import session as session_service`, then call `session_service.get_session(...)`. This avoids name collisions with router functions and makes the delegation explicit.
- Use `BackgroundTasks` for blocking, secondary work in routes
- Prefer Pydantic models over raw dicts for request/response schemas
- ML inference code lives in `ml/`; keep it separate from `services/`. Services call into `ml/` for predictions.
- EEGNet model loaded once via FastAPI lifespan handler; accessed via app state. Never import directly in routes.

## Data Patterns

- Standard PostgreSQL queries on arousal/valence scores (no pgvector, no embedding similarity)
- Synthetic EEG data stored in `backend/data/synthetic/`; DB stores session metadata + segment classifications
- Agent streams responses via Pydantic AI's streaming interface, proxied through a Vercel AI SDK-compatible SSE endpoint
- Agent tools in `agents/tools/`, grouped by capability in `agents/capabilities/`
- Use `scipy` for signal processing (filtering, PSD); use `mne` for advanced EEG analysis — do not mix

## Pydantic

- Prefer Pydantic schemas over dataclasses and raw dicts for data structures
- Use Pydantic v2 conventions: `model_dump()` not `dict()`, `model_validate()` not `parse_obj()`
- Leverage Pydantic features: `@field_validator`, `@model_validator`, `@computed_field`, `Field()` constraints
- Use `model_dump(exclude_unset=True)` for partial updates

## SQLAlchemy

- Prefer simple Python type inference; only use `mapped_column` when column attributes need customization
- Encapsulate DB logic in model `@classmethod` functions: `create`, `update`, `delete` for mutations; `get_by_*` for queries.

## Migrations

- After generating an alembic migration, pause and ask if it looks okay before running `migrate-docker.sh`
- Never run downgrade scripts without explicit user request

## Module Conventions

Re-export convention for `__init__.py`:

- **Default:** Keep `__init__.py` empty; use deep imports (`from cortexdj.models.session import Session`)
- **Exception — `models/`:** Re-export all models for Alembic autogenerate support
- **Exception — `routers/`:** Re-export routers for clean aggregation in `app.py`
