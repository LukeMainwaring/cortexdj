# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

See @README.md for a project overview and @DEVELOPMENT.md for a development guide.

When working with this codebase, prioritize readability over cleverness. Ask clarifying questions before making architectural changes.

## Common Commands

All commands in this file are designed to run from the repo root. Do not use `cd <dir> && ...` patterns -- use `--directory` (uv) or `-C` (pnpm) flags instead.

### Backend (Python)

```bash
# Install dependencies
uv sync --directory backend

# Run backend with Docker (includes PostgreSQL)
docker compose up -d

# Pre-commit hooks (covers type checking, linting, and formatting)
uv run --directory backend pre-commit run --all-files

# Generate synthetic EEG data
uv run --directory backend generate-synthetic

# Train EEGNet model (synthetic data, default)
uv run --directory backend train-model

# Train on DEAP dataset (requires DEAP download — see backend/data/DEAP_SETUP.md)
uv run --directory backend train-model --source deap --model eegnet
uv run --directory backend train-model --source deap --model cbramod --cv loso

# Compare EEGNet vs CBraMod on DEAP
uv run --directory backend compare-models

# Seed database with EEG sessions
uv run --directory backend seed-sessions
uv run --directory backend seed-sessions --source deap

# Create local database migration
./backend/scripts/create-db-revision-docker.sh "<migration_message>"

# Apply pending migrations
./backend/scripts/migrate-docker.sh

# Run tests
uv run --directory backend pytest
```

### Frontend (TypeScript/Next.js)

```bash
pnpm -C frontend install
pnpm -C frontend lint            # lint with ultracite
pnpm -C frontend format          # format with ultracite
pnpm -C frontend generate-client # regenerate API client from backend OpenAPI
```

After making frontend code changes, run `pnpm -C frontend format` to fix formatting. Use `pnpm -C frontend lint` to check for errors.

## Architecture

### Backend (`backend/`)

FastAPI Python backend using async patterns throughout.

- **`src/cortexdj/app.py`**: FastAPI application entry point with lifespan handler (EEGNet model loading)
- **`src/cortexdj/routers/`**: API routes by domain (agent, sessions, threads, health)
- **`src/cortexdj/agents/`**: Pydantic AI agent -- `brain_agent.py` defines the brain assistant with tools for session analysis, brain state insights, playlist curation, Spotify integration, and EEG classification
- **`src/cortexdj/agents/capabilities/`**: Capability classes grouping related tools; ClassificationCapability uses `get_instructions()` to dynamically inject brain context into the system prompt
- **`src/cortexdj/agents/tools/`**: Agent tool implementations (session_tools, insight_tools, playlist_tools, classification_tools)
- **`src/cortexdj/agents/history_processor.py`**: Summarizes large tool results in historical messages to prevent token bloat in multi-turn conversations
- **`src/cortexdj/models/`**: SQLAlchemy async models with CRUD classmethods (Session, EegSegment, Track, SessionTrack, Playlist, Thread, Message)
- **`src/cortexdj/schemas/`**: Pydantic schemas for API contracts
- **`src/cortexdj/services/`**: Business logic (EEG processing, Spotify, session management, thread management, title generation)
- **`src/cortexdj/ml/`**: PyTorch EEGNet with dual classification heads, training script, inference wrapper, EEG preprocessing
- **`src/cortexdj/core/config.py`**: Settings via pydantic-settings
- **`src/cortexdj/migrations/`**: Alembic migrations for PostgreSQL

### Frontend (`frontend/`)

Next.js 16 with App Router, adapted from the SampleSpace project.

- **`app/(chat)/page.tsx`**: Main chat page
- **`app/(chat)/api/chat/route.ts`**: Proxy route to backend agent
- **`components/chat.tsx`**: Chat orchestrator using `@ai-sdk/react` useChat hook
- **`components/brain-context-badge.tsx`**: Displays active brain context (mood/arousal/valence)
- **`components/greeting.tsx`**: CortexDJ-branded empty state

### Data Flow

1. Frontend `useChat` sends messages to `/api/chat` route
2. Route proxies to backend `POST /agent/chat`
3. Backend loads thread's `brain_context` into `AgentDeps`; `ClassificationCapability.get_instructions()` dynamically injects it into the system prompt
4. `HistoryProcessor` summarizes large tool results from prior turns to prevent token bloat
5. Pydantic AI agent decides which tools to call
6. Agent streams response back as SSE (Vercel AI SDK format)
7. Frontend renders with tool-call transparency and brain context badge

## Additional Instructions

- This project uses Pydantic AI for agent orchestration.
- Uses Vercel AI SDK's `useChat` for streaming chat (frontend only).
- Backend port: 8003, Frontend port: 3003, PostgreSQL port: 5433
- Synthetic EEG data is gitignored -- use `uv run generate-synthetic` to create.
- Model checkpoints are gitignored -- use `uv run train-model` to train.
- After modifying backend API endpoints, regenerate the frontend client with `pnpm -C frontend generate-client`.
- Do not manually edit files in `frontend/api/generated/`.
