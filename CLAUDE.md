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

# Train CBraMod model with LOSO CV (default; requires DEAP — see backend/data/DEAP_SETUP.md)
# Labels are binarized at each subject's own Likert median by default; use
# --label-split fixed_5 to reproduce papers that used the >= 5 threshold.
uv run --directory backend train-model

# Quick dev run (10 epochs, 3 folds) — works on MPS
uv run --directory backend train-model --quick

# Train EEGNet instead
uv run --directory backend train-model --model eegnet

# Compare EEGNet vs CBraMod on DEAP (always shows a MajorityBaseline reference row)
uv run --directory backend compare-models

# GPU training via Modal (run `modal setup` once to authenticate)
modal run backend/scripts/modal_train.py

# Seed database with EEG sessions (requires DEAP data)
uv run --directory backend seed-sessions

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
- **`src/cortexdj/ml/`**: PyTorch EEGNet with dual classification heads, training script, inference wrapper, EEG preprocessing. `metrics.py` owns class-weight computation, macro-F1, balanced accuracy, per-class recall, and the `MajorityBaselinePredictor` reference used by `compare-models`. The training loop uses per-fold per-head class-weighted CE with label smoothing and EMA-smoothed early stopping on macro-F1 with a minimum-epochs floor. See `DEVELOPMENT.md` for the `--label-split` options.
- **`src/cortexdj/core/config.py`**: Settings via pydantic-settings
- **`src/cortexdj/migrations/`**: Alembic migrations for PostgreSQL

### Frontend (`frontend/`)

Next.js 16 with App Router, adapted from the SampleSpace project.

- **`app/(chat)/page.tsx`**: Main chat page
- **`app/(chat)/api/chat/route.ts`**: Proxy route to backend agent
- **`components/chat.tsx`**: Chat orchestrator using `@ai-sdk/react` useChat hook
- **`components/brain-context-badge.tsx`**: Displays active brain context (mood/arousal/valence)
- **`components/session-visualization.tsx`**: Tabbed session viewer — wraps `components/emotion-trajectory.tsx` (default, animated SVG trajectory through Russell's affect space) and a recharts arousal/valence timeline in Radix Tabs, with the band-power chart shared below. Auto-rendered by `components/message.tsx` when an `analyze_session` tool call is detected
- **`components/emotion-trajectory.tsx`**: Custom SVG + `motion/react` chart that plots each 4-second segment as a point in the valence/arousal plane, draws a smoothed rolling-mean path via an animated `motion.path` (`style={{ pathLength: progress }}`), and exposes a play/pause + scrubber driven by a `requestAnimationFrame` loop
- **`components/greeting.tsx`**: CortexDJ-branded empty state
- **`api/hooks/sessions.ts`**: TanStack Query wrapper around the generated sessions client (`useSessionSegments`); follow this pattern when wrapping new generated endpoints

### Data Flow

1. Frontend `useChat` sends messages to `/api/chat` route
2. Route proxies to backend `POST /agent/chat`
3. Backend loads thread's `brain_context` into `AgentDeps`; `ClassificationCapability.get_instructions()` dynamically injects it into the system prompt
4. `HistoryProcessor` summarizes large tool results from prior turns to prevent token bloat
5. Pydantic AI agent decides which tools to call
6. Agent streams response back as SSE (Vercel AI SDK format)
7. Frontend renders with tool-call transparency, brain context badge, and inline `<SessionVisualization>` (Trajectory tab default, Timeline tab secondary, band powers below) for `analyze_session` tool calls. The backend `services/trajectory.py` computes a `trajectory_summary` (dwell per quadrant, transitions, centroid, dispersion, path length, smoothed trail) that feeds both the chart and the agent narration; `SessionCapability.get_instructions` tells the agent to cite those fields instead of only the dominant state

## Additional Instructions

- This project uses Pydantic AI for agent orchestration.
- Uses Vercel AI SDK's `useChat` for streaming chat (frontend only).
- Backend port: 8003, Frontend port: 3003, PostgreSQL port: 5433
- Model checkpoints are gitignored -- use `uv run train-model` to train.
- After modifying backend API endpoints, regenerate the frontend client with `pnpm -C frontend generate-client`.
- Do not manually edit files in `frontend/api/generated/`.
- Working with Modal? See `.claude/rules/backend/modal.md` first — Modal's API has changed substantially in the 1.x series and your training data may be stale.
