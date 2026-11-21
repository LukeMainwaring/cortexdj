# AGENTS.md

CortexDJ is a full-stack AI app: a FastAPI + Pydantic AI backend and a Next.js chat frontend. It is being ported onto a fresh instance of its upstream template. Every domain slice is ported and running: **`sessions`** (EEG sessions, trajectories, brain context), the in-repo pattern the other slices copy, **`spotify`** (the Spotify login, track search, the user's library, mood playlists), and **`retrieval`** (tracks matched to a session's EEG through a pgvector index of CLAP audio vectors).

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

# Seed the EEG sessions from the 32 DEAP files in backend/data/deap (several minutes;
# needs Postgres up and migrated; --participants 1 2 3 seeds a subset)
uv run --directory backend seed-sessions

# Restore the track index (447 CLAP vectors) into a migrated database. seed-track-index
# cannot build it today; DEVELOPMENT.md says why and what the dump is.
docker compose exec -T postgres pg_restore -U postgres -d cortexdj --data-only < backend/data/track-index.dump
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
- **`src/cortexdj/shared/`**: cross-feature primitives — `model_base.py` (`Base`), `schema_base.py` (`BaseSchema`), and the seams one slice offers the others: `track_access.py` (find tracks by mood, read tracks, save a track's Spotify id) and `session_lookup.py` (a session's participant id), both implemented by `sessions` and registered in `api.py`. The `spotify` slice reaches tracks only through `track_access.py`, and `retrieval` reads a session's participant only through `session_lookup.py`.
- **`src/cortexdj/agents/`**: the feature-agnostic Pydantic AI **agent runtime** — `assistant.py` holds `build_agent`, the shared recipe (hooks, history compaction, prompt caching, optional thinking, tool timeout) and builds the assistant from `ENABLED_CAPABILITIES` (the capability registry in `capabilities.py`); `registry.py` is the **named-agent registry** (`get_agent(ASSISTANT)`, `get_agent(THREAD_TITLE)`) that builds lazily and caches; plus `title_agent.py`, `deps.py`, `hooks.py`, `history_processor.py`. Convention: tools and output validators let exceptions propagate to `hooks.on_tool_execute_error` / `hooks.on_output_process_error` for structured recovery rather than catching inline.
- **`src/cortexdj/ml/`**: the **ML inference kernel** for the two EEG pipelines, the quadrant classifier and the contrastive EEG↔CLAP encoder — models, datasets, preprocessing, metrics, `predict.py`, `pretrained.py`, `device.py`. Training code is not in this package.
- **`src/cortexdj/features/<name>/`**: one vertical slice per domain — `model.py`, `schema.py`, `service.py`, `router.py`, and (if it talks to the agent) `tools.py` + `capability.py`. **Copy a slice to add a domain.** Handlers stay thin and delegate to `service.py`. The slices are `chat` (threads, messages, title generation, the streaming `/agent/chat` endpoint), `sessions`, `spotify`, and `retrieval`.
- **`src/cortexdj/features/sessions/`**: EEG sessions, segments, tracks, and a thread's brain context. Beyond the role files it has `trajectory.py` (the affect-path analytics), `emotion.py`, and `eeg_model.py`, the **EEG model holder**: a startup hook loads the `EEG_MODEL_BACKEND` checkpoint into it, a missing checkpoint is a startup warning, and only `get_model_info` and the rule that hides it read it (`override_eeg_model` in tests). Brain context is the slice's own table keyed by `thread_id` with a string foreign key to `threads`; the capability reads it from the database on every run and `GET /brain-context/{thread_id}` serves it to the frontend.
- **`src/cortexdj/features/spotify/`**: the Spotify OAuth routes (`/spotify/connect`, `/callback`, `/status`, `/disconnect`), the single-row `spotify_tokens` table, locally saved `playlists`, and the assistant's Spotify tools. Beyond the role files it has `client.py`, the **Spotify client getter**: `get_spotify_client(db)` returns the user's client when a token row exists (refreshing a token about to expire) and otherwise the app's client-credentials client, with `is_user` saying which and `client` None when `SPOTIFY_CLIENT_ID`/`SPOTIFY_CLIENT_SECRET` are unset. `AgentDeps` carries no client: the tools and the capability's `prepare_tools` call the getter with `ctx.deps.db`, one token-row read per model step, and the four library tools are hidden unless `is_user` (`override_spotify_client` in tests). Without the user's login `build_mood_playlist` saves the playlist locally only; it still needs the app client to resolve the tracks' Spotify ids, so with Spotify unconfigured it fails. The callback path `/api/spotify/callback` must match `SPOTIFY_REDIRECT_URI` and the Spotify app's registered redirect URI, and it redirects the browser to `{FRONTEND_URL}/settings/spotify`.
- **`src/cortexdj/features/retrieval/`**: brain-to-track retrieval. `GET /sessions/{session_id}/similar-tracks` and the `retrieve_tracks_from_brain_state` tool embed a session's raw EEG (re-read from the participant's DEAP file, found through the session-lookup seam) and rank the rows of `track_audio_embeddings`, 512-dimension CLAP audio vectors, by cosine distance; `GET /audio/preview/{cache_key}` serves a cached iTunes preview for the panel's player. Beyond the role files it has `encoder.py`, the **contrastive encoder holder**: a startup hook loads `contrastive_best.pt` into it, a missing checkpoint is a startup warning, the route reads it through a FastAPI dependency and the tool through `get_contrastive_encoder` and both pass it to the service, and with none loaded retrieval answers 503 or a `ToolFailed` that points at the checkpoint download script (`override_contrastive_encoder` in tests). `audio_catalog.py` resolves a Spotify track to an iTunes preview for the scripts. pgvector is in the template's form: the initial migration creates the `vector` extension, and the HNSW index (`vector_cosine_ops`, `m=16`, `ef_construction=64`) is declared on the model and created with `op.create_index`, never raw SQL.
- **`src/cortexdj/scripts/`**: command-line entry points declared under `[project.scripts]`: `seed-sessions`, `seed-track-index` (builds the track index; it cannot finish until `.scratch/audio-decoding/issues/01-decode-m4a-previews.md` is done, so the index is restored from `backend/data/track-index.dump`), and `fetch-deap-audio` (previews for the 40 DEAP stimuli). The last two get their Spotify client from `get_spotify_client(db)` and each deletes its miss log under `backend/data/` as its first step, so run one only when asked. Outside `features/` and the kernel, so a script may import across features; it loads what it needs itself rather than through the app's startup hooks.
- **`src/cortexdj/migrations/`**: Alembic migrations for PostgreSQL. `env.py` does `import cortexdj.features` so autogenerate sees every slice's tables.

**Dependency direction:** a feature may import `core/`, `shared/`, `agents/`, `ml/`; the kernel never imports a feature; `ml/` imports only `core/`; a feature never imports another feature. Only `api.py`, `agents/capabilities.py`, and `agents/registry.py` import across features. When a feature or the runtime needs something another feature owns, the type and a one-slot `register_<thing>` / `override_<thing>` seam go in `shared/`, `api.py` registers the implementation at import, and a call before registration raises a `<Thing>NotRegistered` error. Never a direct import.

### Frontend (`frontend/`)

Next.js with the App Router. The generated API client and TanStack Query wrappers keep the UI in sync with the backend's OpenAPI schema.

- **`features/<name>/`**: domain UI slices, mirroring the backend's `features/<name>/` (`features/sessions/` is the example; `features/retrieval/` holds the retrieved-tracks panel and its preview player). A slice's tool panels register in **`features/tool-panel-registry.ts`**, the frontend analog of `ENABLED_CAPABILITIES`, and what a slice shows in the chat header (the brain-context badge) registers in **`features/header-slot-registry.ts`**; the shell reaches a slice only through those two files. Everything else (`app/`, `api/`, `components/ui` + `components/elements`, the chat shell, `lib/`, `hooks/`) is the shared kernel. The Spotify settings page (`app/(chat)/settings/spotify/`, reached from the sidebar's settings menu) and its hooks (`api/hooks/spotify.ts`) are kernel too: the Spotify tools have no panels, so there is no frontend `spotify` slice.
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
