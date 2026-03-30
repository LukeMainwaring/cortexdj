# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

See @README.md for a project overview and @DEVELOPMENT.md for a development guide.

When working with this codebase, prioritize readability over cleverness. Ask clarifying questions before making architectural changes.

## Common Commands

### Backend (Python)

```bash
# Install dependencies
cd backend && uv sync

# Run backend with Docker (includes PostgreSQL)
docker compose up -d

# Pre-commit hooks (covers type checking, linting, and formatting)
uv run pre-commit run --all-files

# Generate synthetic EEG data
uv run generate-synthetic

# Train EEGNet model
uv run train-model

# Seed database with EEG sessions
uv run seed-sessions

# Create local database migration
cd backend && ./scripts/create-db-revision-docker.sh "<migration_message>"

# Apply pending migrations
cd backend && ./scripts/migrate-docker.sh
```

### Frontend (TypeScript/Next.js)

```bash
cd frontend && pnpm install
cd frontend && pnpm lint            # lint with ultracite
cd frontend && pnpm format          # format with ultracite
cd frontend && pnpm generate-client # regenerate API client from backend OpenAPI
```

After making frontend code changes, run `pnpm format` to fix formatting. Use `pnpm lint` to check for errors.

## Architecture

### Backend (`backend/`)

FastAPI Python backend using async patterns throughout.

- **`src/cortexdj/app.py`**: FastAPI application entry point with lifespan handler (EEGNet model loading)
- **`src/cortexdj/routers/`**: API routes by domain (agent, sessions, threads, health)
- **`src/cortexdj/agents/`**: Pydantic AI agent -- `brain_agent.py` defines the brain assistant with tools for session analysis, brain state insights, playlist curation, and EEG classification
- **`src/cortexdj/agents/capabilities/`**: Capability classes grouping related tools (SessionCapability, InsightCapability, PlaylistCapability, ClassificationCapability)
- **`src/cortexdj/agents/tools/`**: Agent tool implementations (session_tools, insight_tools, playlist_tools, classification_tools)
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
3. Backend loads thread's `brain_context` and injects into `AgentDeps`
4. Pydantic AI agent decides which tools to call
5. Agent streams response back as SSE (Vercel AI SDK format)
6. Frontend renders with tool-call transparency and brain context badge

## Additional Instructions

- This project uses Pydantic AI for agent orchestration.
- Uses Vercel AI SDK's `useChat` for streaming chat (frontend only).
- Backend port: 8003, Frontend port: 3003, PostgreSQL port: 5433
- Synthetic EEG data is gitignored -- use `uv run generate-synthetic` to create.
- Model checkpoints are gitignored -- use `uv run train-model` to train.
- After modifying backend API endpoints, regenerate the frontend client with `cd frontend && pnpm generate-client`.
- Do not manually edit files in `frontend/api/generated/`.
