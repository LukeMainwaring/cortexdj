# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

For architecture and project overview, read `README.md`. For setup, commands, training workflows, and database migrations, read `DEVELOPMENT.md`.

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

# GPU training via Modal (run `modal setup` once to authenticate)
modal run backend/scripts/modal_train.py

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

- **`src/cortexdj/app.py`**: FastAPI entry point with lifespan handler (EEG model loading)
- **`src/cortexdj/routers/`**: API routes by domain (agent, sessions, threads, health, retrieval)
- **`src/cortexdj/agents/`**: Pydantic AI agent — `brain_agent.py` is the assistant. Capabilities in `capabilities/` (Session, Insight, Playlist, Retrieval, Classification) group tools and optionally inject dynamic system-prompt fragments via `get_instructions()`. Tool functions live in `tools/`. Key convention: tools let exceptions propagate to `hooks.on_tool_execute_error` for structured recovery, except `retrieval_tools.retrieve_tracks_from_brain_state` which catches `DeapFileMissingError` inline for user-facing clarity. `history_processor.py` summarizes large tool results in prior turns to prevent token bloat.
- **`src/cortexdj/models/`**: SQLAlchemy async models with CRUD classmethods (Session, EegSegment, Track, SessionTrack, Playlist, Thread, Message, TrackAudioEmbedding). `TrackAudioEmbedding` is the pgvector-backed retrieval index with an HNSW cosine search op.
- **`src/cortexdj/schemas/`**: Pydantic schemas for API contracts
- **`src/cortexdj/services/`**: Business logic — `eeg_processing`, `spotify` (identity only; `preview_url` deprecated for standard-mode apps Nov 2024), `session`, `thread`, `title_generator`, `trajectory`, `retrieval` (encode_session_to_clap_space + pgvector search), `audio_catalog` (iTunes Search wrapper with duration-delta match heuristic).
- **`src/cortexdj/ml/`**: Two separable model pipelines.
  - **Quadrant classifier**: `model.py` (EEGNet), `pretrained.py` (CBraMod), `dataset.py`, `preprocessing.py`, `train.py`, `metrics.py`, `predict.py`. Per-fold class-weighted CE + label smoothing, EMA-smoothed early stop on macro-F1. See `DEVELOPMENT.md` for `--label-split` options.
  - **Contrastive EEG↔CLAP**: `contrastive.py` (EegCLAPEncoder with CBraMod backbone + SimCLR projection head, symmetric soft-target InfoNCE, encode_session), `contrastive_dataset.py` (DeapClapPairDataset + host-portable audio cache + `trial_to_eeg_windows` shared slicer), `contrastive_train.py` (SequentialLR warmup+cosine, TensorBoard scalars + val embedding projector, grad accumulation).
- **`src/cortexdj/core/config.py`**: Settings via pydantic-settings
- **`src/cortexdj/migrations/`**: Alembic migrations for PostgreSQL

### Frontend (`frontend/`)

Next.js 16 with App Router.

- **`app/(chat)/page.tsx`**: Main chat page
- **`app/(chat)/api/chat/route.ts`**: Proxy route to backend agent
- **`components/chat.tsx`**: Chat orchestrator using `@ai-sdk/react` useChat hook
- **`components/brain-context-badge.tsx`**: Displays active brain context (mood/arousal/valence)
- **`components/session-visualization.tsx`**: Tabbed session viewer — wraps `components/emotion-trajectory.tsx` (default, animated SVG trajectory through Russell's affect space) and a recharts arousal/valence timeline in Radix Tabs, with the band-power chart shared below. Auto-rendered by `components/message.tsx` when an `analyze_session` tool call is detected
- **`components/emotion-trajectory.tsx`**: Custom SVG + `motion/react` chart that plots each 4-second segment as a point in the valence/arousal plane, draws a smoothed rolling-mean path via an animated `motion.path` (`style={{ pathLength: progress }}`), and exposes a play/pause + scrubber driven by a `requestAnimationFrame` loop
- **`components/retrieved-tracks-panel.tsx`**: Ranked tracks rendered beneath `retrieve_tracks_from_brain_state` tool calls — similarity bars, inline 30s preview playback via a shared `<audio>` ref, Spotify deep-links, branched 404/503/500 error states
- **`api/hooks/sessions.ts`**: TanStack Query wrappers around the generated sessions client — `useSessionSegments` and `useSimilarTracks`; follow this pattern when wrapping new generated endpoints. Retries skip 404 (missing session) and 503 (missing contrastive checkpoint)

### Data Flow

1. Frontend `useChat` sends messages to `/api/chat` route
2. Route proxies to backend `POST /agent/chat`
3. Backend loads thread's `brain_context` into `AgentDeps`; `ClassificationCapability.get_instructions()` dynamically injects it into the system prompt
4. `HistoryProcessor` summarizes large tool results from prior turns to prevent token bloat
5. Pydantic AI agent decides which tools to call
6. Agent streams response back as SSE (Vercel AI SDK format)
7. Frontend renders with tool-call transparency, brain context badge, and inline panels: `<SessionVisualization>` on `analyze_session`, `<RetrievedTracksPanel>` on `retrieve_tracks_from_brain_state` (component-level details in the Frontend section above)

## Additional Instructions

- Backend port: 8003, Frontend port: 3003, PostgreSQL port: 5433
- Model checkpoints are gitignored -- use `uv run train-model` to train.
- After modifying backend API endpoints, regenerate the frontend client with `pnpm -C frontend generate-client`.
- Do not manually edit files in `frontend/api/generated/`.
- Working with Modal? See `.claude/rules/backend/modal.md` first — Modal's API has changed substantially in the 1.x series and your training data may be stale.
