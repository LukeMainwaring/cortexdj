---
paths:
  - "backend/**/*.py"
---

# Backend Patterns

Python/FastAPI conventions for the CortexDJ backend.

## Code Style

- Use lowercase with underscores for filenames (e.g., `brain_agent.py`, `eeg_processing.py`)
- Use modern Python syntax: `| None` over `Optional`, `list` over `List`
- Use f-strings for logging: `logger.info(f"Created {item.id}")`
- Use descriptive variable names with auxiliary verbs (e.g., `is_active`, `has_permission`)
- Type hints required on all functions
- **Comments and docstrings:** Only add comments that explain *why*, not *what*. Don't add docstrings that restate the function name (e.g., `"""Delete a document."""` on `delete()`). Don't add `Args:`/`Returns:` sections that duplicate type annotations. Router endpoint docstrings are the exception — keep those since FastAPI surfaces them in OpenAPI docs. When a docstring adds genuine value (non-obvious behavior, important caveats), keep it concise — one or two lines.

## Architecture

- Use `def` for pure functions, `async def` for I/O operations
- Use FastAPI's dependency injection for shared resources (db sessions, ML models)
- All database operations are async using `AsyncSession`
- Keep route handlers thin: push business logic to `services/`, DB logic to `models/`
- Import service modules with a named alias in routers: `from cortexdj.services import session as session_service`, then call `session_service.get_session(...)`. This avoids name collisions with router functions and makes the delegation explicit.
- Use `BackgroundTasks` for blocking, secondary work in routes
- Prefer Pydantic models over raw dicts for request/response schemas
- ML inference code lives in `ml/`; keep it separate from `services/`. Services call into `ml/` for predictions.
- EEGNet model loaded once via FastAPI lifespan handler; accessed via typed dependency injection (`EEGModelDep` in `dependencies/eeg_model.py`). Never import directly in routes.
- Capabilities can use `get_instructions()` to dynamically inject runtime context into the agent system prompt (e.g., `ClassificationCapability` injects active brain context). Use sync functions for non-I/O injections.

## Data Patterns

- Arousal/valence scores are plain columns queried with standard SQL; embedding similarity lives in `TrackAudioEmbedding` (pgvector, HNSW cosine search) and is reached through the retrieval service — don't add ad-hoc vector queries elsewhere
- DEAP EEG data stored in `backend/data/deap/`; DB stores session metadata + segment classifications
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

## Testing

- Three tiers under `backend/tests/`: `unit/<area>/` (fast, DB-free, provider-free — the default `pytest` run), `integration/` (real Postgres + HTTP, opt-in via `-m integration`), and `evals/` (real-LLM, `@pytest.mark.eval`, opt-in via `-m eval`). The default run excludes both opt-in tiers through `addopts = "-m 'not eval and not integration'"`; test dirs are packages so basenames can repeat across tiers.
- Async tests use anyio's pytest plugin: `@pytest.mark.anyio` plus the `anyio_backend` fixture in the root conftest. No pytest-asyncio, no raw `asyncio.run()` wrappers.
- The integration tier builds schema from the shipped Alembic migrations and rolls back an outer transaction per test. Isolation holds only because the app's sole commit point is the session dependency (`dependencies/db.py`) — services and model classmethods `flush()`, never `commit()`. The tier refuses database names not ending in `_test` and applies its markers structurally in `tests/integration/conftest.py` (don't add per-file `pytestmark`). Local setup: `backend/scripts/create-test-db.sh`, then `POSTGRES_DB=cortexdj_test uv run --directory backend pytest -m integration`.
- Shared builders live in `tests/factories.py` (`make_*` = unpersisted, `create_*` = persisted via flush); agent-deps mocks in `tests/fakes.py`.
- Root `conftest.py` sets a dummy `OPENAI_API_KEY` so importing `brain_agent` never needs a real key; tests that invoke the agent use `agent.override` with pydantic-ai `TestModel`.
- CI's test job enforces the coverage floor; don't run coverage locally unless you're investigating a CI failure.

## Dependencies

- Never hand-edit `pyproject.toml` with a remembered version — models pin stale releases from training memory. Run `uv add <package> --directory backend` (add `--dev` for tooling) so the resolver fetches the current compatible release and updates `uv.lock` in one step.
- Don't pin an exact `==` version unless there's a specific reason; let `uv add` write the default lower-bound constraint.

## Module Conventions

Re-export convention for `__init__.py`:

- **Default:** Keep `__init__.py` empty; use deep imports (`from cortexdj.models.session import Session`)
- **Exception — `models/`:** Re-export all models for Alembic autogenerate support
- **Exception — `routers/`:** Re-export routers for clean aggregation in `app.py`
