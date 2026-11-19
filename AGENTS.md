# AGENTS.md

CortexDJ is a full-stack AI app: a FastAPI + Pydantic AI backend and a Next.js chat frontend. It is being ported onto a fresh instance of its upstream template. Until the port finishes, the running app is the template's example domain, **Notes** — a vertical feature slice you copy as the pattern when a domain is ported in.

**`legacy/`** holds the pre-port source (`legacy/backend/`, `legacy/frontend/`, `legacy/docs/`, `legacy/DEVELOPMENT.md`). Nothing in it runs and every check excludes it. Code leaves it with `git mv` as it is ported, so file history follows; do not edit, lint, or test it in place. `README.md` still describes the pre-port app. The pre-port tree is also readable at the `pre-port` tag.

For project overview and quickstart, read `README.md`. For setup, commands, database migrations, and tests, read `DEVELOPMENT.md`.

When working with this codebase, prioritize readability over cleverness. Ask clarifying questions before making architectural changes.

## Common Commands

All commands in this file are designed to run from the repo root. Do not use `cd <dir> && ...` patterns -- use `--directory` (uv) or `-C` (pnpm) flags instead.

### Backend (Python)

```bash
# Install dependencies
uv sync --directory backend

# Run backend with Docker (includes PostgreSQL)
docker compose up -d

# Create local database migration
./backend/scripts/create-db-revision-docker.sh "<migration_message>"

# Apply pending migrations
./backend/scripts/migrate-docker.sh
```

Checks, lightest first. While working, run the first two; before finishing, run the last three once each.

```bash
# One test file
uv run --directory backend pytest tests/unit/<path>

# Type check (mypy strict)
uv run --directory backend mypy src tests

# Unit tier (fast, no DB)
uv run --directory backend pytest

# Unit + integration against the test database. Starts Postgres if it is not
# running and stops it again if it started it; extra args go to pytest.
./backend/scripts/test-integration.sh

# Pre-commit hooks (lint, format, mypy, import linter)
uv run --directory backend pre-commit run --all-files
```

The eval tier (`backend/tests/evals/`) is not one of these checks: it calls real models and costs money. It needs Postgres and the provider keys in `.env`; run it with `./backend/scripts/run-evals.sh`, and only when the user asks.

Tests are how a change is verified. Start the app stack (`docker compose up -d`) only when the user asks for it, and stop it with `docker compose stop` when done.

### Frontend (TypeScript/Next.js)

```bash
pnpm -C frontend install
pnpm -C frontend lint            # lint with ultracite
pnpm -C frontend format          # format with ultracite
pnpm -C frontend typecheck       # tsc, including the generated client's consumers
pnpm -C frontend generate-client # regenerate API client from backend OpenAPI
```

After making frontend code changes, run `pnpm -C frontend format` to fix formatting. Use `pnpm -C frontend lint` to check for errors.

## Architecture

### Backend (`backend/`)

FastAPI Python backend (package `cortexdj`), async throughout, organized **feature-first**: each domain is a vertical slice under `features/<name>/`, with a thin horizontal kernel (`core/`, `shared/`, `agents/`, `ml/`) holding what every feature shares. New features, routers, and startup hooks plug into small registries rather than edits scattered across `app.py` — register in one place and the wiring follows.

- **`src/cortexdj/app.py`**: FastAPI entry point with a lifespan handler that runs the startup-hook registry. Don't add ad-hoc startup logic here — register a hook (see `core/startup.py`).
- **`src/cortexdj/api.py`**: the **router aggregator** — include a feature's router here and it's mounted. A composition root (may import across features).
- **`src/cortexdj/core/`**: platform kernel — `config.py` (settings), `db.py`, `startup.py` (the **startup-hook registry**), `logging.py`, `health.py`, `paths.py` (the data paths under `backend/data/`). Never cut by a feature prune.
- **`src/cortexdj/shared/`**: cross-feature primitives — `model_base.py` (`Base`), `schema_base.py` (`BaseSchema`).
- **`src/cortexdj/agents/`**: the feature-agnostic Pydantic AI **agent runtime** — `assistant.py` holds `build_agent`, the shared recipe (hooks, history compaction, prompt caching, optional thinking, tool timeout) and builds the assistant from `ENABLED_CAPABILITIES` (the capability registry in `capabilities.py`); `registry.py` is the **named-agent registry** (`get_agent(ASSISTANT)`, `get_agent(THREAD_TITLE)`) that builds lazily and caches; plus `title_agent.py`, `deps.py`, `hooks.py`, `history_processor.py`. Convention: tools and output validators let exceptions propagate to `hooks.on_tool_execute_error` / `hooks.on_output_process_error` for structured recovery rather than catching inline.
- **`src/cortexdj/ml/`**: the **ML inference kernel** for the two EEG pipelines, the quadrant classifier and the contrastive EEG↔CLAP encoder — models, datasets, preprocessing, metrics, `predict.py`, `pretrained.py`, `device.py`. Training code is not in this package.
- **`src/cortexdj/features/<name>/`**: one vertical slice per domain — `model.py`, `schema.py`, `service.py`, `router.py`, and (if it talks to the agent) `tools.py` + `capability.py`. **Copy a slice to add a domain.** Handlers stay thin and delegate to `service.py`. Ships with `notes` and `chat` (threads, messages, title generation, the streaming `/agent/chat` endpoint).
- **`src/cortexdj/migrations/`**: Alembic migrations for PostgreSQL. `env.py` does `import cortexdj.features` so autogenerate sees every slice's tables.

**Dependency direction:** a feature may import `core/`, `shared/`, `agents/`, `ml/`; the kernel never imports a feature; `ml/` imports only `core/`; a feature never imports another feature. Only `api.py`, `agents/capabilities.py`, and `agents/registry.py` import across features. When a feature or the runtime needs something another feature owns, the type and a one-slot `register_<thing>` / `override_<thing>` seam go in `shared/`, `api.py` registers the implementation at import, and a call before registration raises a `<Thing>NotRegistered` error. Never a direct import.

### Frontend (`frontend/`)

Next.js with the App Router. The generated API client and TanStack Query wrappers keep the UI in sync with the backend's OpenAPI schema.

- **`features/<name>/`**: domain UI slices, mirroring the backend's `features/<name>/` (the example is `features/notes/`). A slice's tool panels register in **`features/tool-panel-registry.ts`**, the frontend analog of `ENABLED_CAPABILITIES`. Everything else (`app/`, `api/`, `components/ui` + `components/elements`, the chat shell, `lib/`, `hooks/`) is the shared kernel.
- **`api/hooks/`**: TanStack Query wrappers around the generated client (e.g. `threads.ts`); follow this pattern when wrapping new generated endpoints.
- **`api/generated/`**: generated from the backend OpenAPI schema via `pnpm -C frontend generate-client`.

### Data Flow

A chat turn, from the browser's `useChat` through the `/api/chat` proxy, the backend's `POST /agent/chat`, the agent run, and the SSE stream back to the tool panels, is described in `docs/architecture/one-chat-turn.md`. The thread id reaches tools as `ctx.conversation_id`; capabilities inject runtime context into the system prompt via `get_instructions()`.

## Additional Instructions

- Ports: backend 8003, frontend 3003, PostgreSQL 5433 on the host (5432 inside its container and in CI).
- After modifying backend API endpoints, regenerate the frontend client with `pnpm -C frontend generate-client`. Never hand-edit `frontend/api/generated/`.
- Before writing code, read the rules for the files you touch (Claude Code loads them by path through the `.claude/rules` symlink):
  - Any backend or frontend code: `.agents/rules/conventions.md` (docstrings that carry the why, current state not history).
  - Backend Python: `.agents/rules/backend/code-conventions.md` (copy targets, test seams, Pydantic vs dataclass); agents, tools, capabilities, or evals also: `.agents/rules/backend/pydantic-ai.md` (doc sources, cached instructions, output tools).
  - Frontend TypeScript: `.agents/rules/frontend/code-conventions.md` (adding shadcn components, query hooks); chat UI or streaming also: `.agents/rules/frontend/vercel-ai-sdk.md` (bundled AI SDK docs; backend SSE is the source of truth).
- When asked to review changes (a diff, branch, or PR): read `REVIEW.md` and apply it. The `review-changes` skill wraps the harness reviewer for a solo or cross-model review; don't add a custom reviewer agent.

## Pull requests

Branch from a freshly pulled `main` and open the PR against it as a draft, with the body from the `pr` skill. When the diff changes code, recommend `review-changes`; otherwise say why no review is needed. When more commits land on an open PR, rewrite its title and body to describe the PR as it now stands, not as a log of what was pushed. Merge only on the user's explicit approval: mark it ready, wait for CI, merge with `--merge`, and clean up as `docs/agents/issue-tracker.md` says.

## Agent skills

### Issue tracker

Issues and specs live as local markdown under `.scratch/<effort>/` in this repo, and `.scratch/active.md` links the unresolved tickets. `docs/agents/issue-tracker.md` is the workflow contract: layout, status, readiness, the planning handoff, and resolving and cleaning up at merge. Consult it when planning ends, when creating or picking up tickets, and when the user approves a merge. Planning sessions (grill-me, grill-with-docs, wayfinder) end with settled decisions and remaining uncertainties, then stop for the user to invoke `to-spec`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`, which also holds the house ADR shape (stricter than `domain-modeling`'s).

### Architecture guide

`docs/architecture/` is the architecture guide: an overview plus one page per topic, each with a diagram and how to explain it. The user-invoked `architecture-guide` skill writes and updates it (`check` lists stale pages).
