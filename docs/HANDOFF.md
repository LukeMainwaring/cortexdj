# CortexDJ — Session Handoff

## What Was Built

CortexDJ was scaffolded from scratch in `~/projects/cortexdj` as a new git repo (not yet committed). The entire codebase was modeled after the samplespace project's architectural patterns, with Spotify integration patterns adapted from the earworm project.

### Backend (Python — 35+ source files)

| Layer | Files | Status |
|-------|-------|--------|
| **Scaffolding** | `pyproject.toml`, `Dockerfile`, `docker-compose.yml`, `.env.sample`, `.pre-commit-config.yaml`, shell scripts | Complete |
| **Config** | `core/config.py` (pydantic-settings with Postgres, Spotify, Agent settings) | Complete |
| **Database** | `dependencies/db.py` (async SQLAlchemy + psycopg), `models/` (8 models), `schemas/` (8 schema files) | Complete, **migration not yet generated** |
| **ML Pipeline** | `ml/preprocessing.py` (bandpass filter, DE features, band powers), `ml/dataset.py` (EEGEmotionDataset), `ml/model.py` (EEGNetClassifier dual-head), `ml/train.py` (5-fold CV), `ml/predict.py` (inference wrapper) | Complete, **not yet tested** |
| **Services** | `services/session.py`, `services/eeg_processing.py`, `services/spotify.py`, `services/thread.py`, `services/title_generator.py` | Complete |
| **Agent** | `agents/brain_agent.py` (system prompt + 4 capabilities), `agents/deps.py`, 4 capability files, 4 tool files (9 total tools) | Complete |
| **Routers** | `routers/agent.py` (SSE streaming), `routers/sessions.py`, `routers/thread.py`, `routers/health.py`, `routers/main.py` | Complete |
| **App** | `app.py` (lifespan loads EEGNet, CORS, logging middleware) | Complete |
| **Scripts** | `scripts/generate_synthetic.py`, `scripts/seed_sessions.py` | Complete |
| **Utils** | `utils/emotion.py`, `utils/logging.py`, `utils/message_serialization.py` | Complete |

### Frontend (TypeScript — adapted from samplespace)

Copied from samplespace and modified:
- Removed: sample-browser, candidate-samples, waveform-viz, audio-block, kit-block, pair-verdict-block, sample-card, upload hooks, wavesurfer dependency
- Added: `brain-context-badge.tsx` (replaces song-context-badge)
- Modified: `chat.tsx` (removed upload flow), `chat-header.tsx` (brain context), `multimodal-input.tsx` (no file attachments), `greeting.tsx` (CortexDJ branding), `app-sidebar.tsx` (CortexDJ logo), `sidebar-user-nav.tsx` (removed sample library links), `tool-call.tsx` (CortexDJ tool verbs), `response.tsx` (removed custom language renderers), `api/client.ts` (port 8003), `api/hooks/threads.ts` (`useThreadBrainContext` replaces `useThreadSongContext`)
- Ports: dev server on 3003, backend URL points to 8003

### Documentation & Config

- `README.md` — Full project overview with Mermaid architecture diagram, tech stack table, setup guide, EEG pipeline diagram
- `DEVELOPMENT.md` — Developer guide with all commands
- `CLAUDE.md` — Claude Code project instructions
- `docs/ROADMAP.md` — 6-phase roadmap (real datasets, BCI, advanced ML, Spotify deep integration, visualization, platform)
- `docs/feature-brainstorm.md` — Extended feature ideas
- `docs/pydantic-ai-llms-full.txt` — Pydantic AI reference (downloaded)
- `docs/vercel-ai-sdk-ui.txt` — Vercel AI SDK UI reference (downloaded)
- `.claude/settings.json` — Permissions config
- `.claude/rules/backend/code-conventions.md` — Backend coding conventions
- `.github/workflows/ci.yml` — CI pipeline (pre-commit on push/PR)

---

## Current State

- **Git**: Initialized but **no commits yet**
- **Backend deps**: `uv sync` completed, `.venv` and `uv.lock` exist
- **Frontend deps**: `pnpm install` completed, `node_modules` and `pnpm-lock.yaml` exist
- **Database**: No migration generated yet (models are defined but no Alembic version file)
- **Synthetic data**: Not yet generated (need `uv run generate-synthetic`)
- **Model**: Not yet trained (need `uv run train-model`)
- **API client**: Not yet generated (need backend running first, then `pnpm generate-client`)

---

## Action Items (in order)

### 1. Initial commit
```bash
cd ~/projects/cortexdj
git add -A
git commit -m "Initial CortexDJ scaffolding — full-stack EEG classifier + Spotify curator"
```

### 2. Get the backend running
```bash
# Start PostgreSQL
docker compose up postgres -d

# Generate the initial Alembic migration
cd backend && ./scripts/create-db-revision-docker.sh "initial schema"
# Review the migration, then apply:
./scripts/migrate-docker.sh

# Start backend (can use Docker or run directly)
docker compose up backend -d
# OR for local dev with hot reload:
cd backend && uv run python -m uvicorn cortexdj.app:app --host 127.0.0.1 --port 8003 --reload
```

### 3. Generate synthetic data + train model + seed DB
```bash
cd backend
uv run generate-synthetic           # Creates data/synthetic/s01.npz through s32.npz
uv run train-model                  # Trains EEGNet, saves data/checkpoints/eegnet_best.pt
uv run seed-sessions                # Populates sessions, segments, tracks, session_tracks
```

### 4. Get the frontend running
```bash
cd frontend
pnpm generate-client                # Fetch OpenAPI spec from running backend, generate types
pnpm dev                            # Start on localhost:3003
```

### 5. Test the chat flow
Open http://localhost:3003 and try:
- "Show me my EEG sessions"
- "Analyze session [paste an ID from the list]"
- "What was my brain doing during that session?"
- "Build me a relaxation playlist"
- "Compare two sessions" (give it two IDs)

### 6. Run pre-commit checks
```bash
cd backend && uv run pre-commit run --all-files
```
This will likely surface type errors or import issues that need fixing. The code was written without running mypy, so expect some `type: ignore` additions or signature fixes.

### 7. Fix likely issues

**Known things to watch for:**
- **mypy strict mode** — Some numpy type annotations in `ml/preprocessing.py` use verbose generic syntax that mypy may flag. May need to simplify to `npt.NDArray[np.floating[Any]]` or add overrides.
- **Alembic migration** — The autogenerate should pick up all 8 tables. Verify the migration looks correct before applying.
- **Frontend import errors** — Some samplespace imports may linger in copied files (e.g., references to `useThreadSongContext` in components we didn't modify, or imports of deleted components). Run `pnpm lint` to find them.
- **`chat-actions-provider.tsx`** — Still imported in `chat.tsx` but may reference samplespace-specific types. Check it compiles.
- **Generated API client** — The backend must be running with the `/api/openapi.json` endpoint accessible before `pnpm generate-client` will work.
- **Spotify tools** — `get_listening_history` tool uses `search_tracks` as a placeholder (real listening history requires user OAuth, which is a roadmap item). Works but returns search results, not actual history.
- **`pydantic-settings`** dependency — Ensure `pydantic-settings` is in `pyproject.toml` (it is, listed as a transitive dep of `pydantic-ai`, but if mypy complains, add it explicitly).

---

## Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| **No pgvector** | EEG data queried by arousal/valence scores, not embedding similarity |
| **postgres:17** (not pgvector image) | Simpler, lighter |
| **Ports 8003/3003/5433** | Avoid conflicts with samplespace (8002/3002/5432) |
| **Spotify optional** | `prepare_tools` hides Spotify-dependent tools when unconfigured |
| **Synthetic data for MVP** | Avoids DEAP dataset registration; realistic enough to demo the full pipeline |
| **EEGNet dual-head** | Custom PyTorch module wrapping spatial/temporal convolutions, not just importing braindecode |
| **Thread brain_context** | Same JSONB merge-on-update pattern as samplespace's song_context |

## Reference Codebases

- **samplespace** (`~/projects/samplespace`) — Primary template for FastAPI + Pydantic AI + Next.js patterns
- **earworm** (`~/projects/earworm`) — Spotify OAuth flow, async spotipy wrapper, playlist tools, `prepare_tools` capability pattern
- **EEG_classifier_spotify** (GitHub: `LukeMainwaring/EEG_classifier_spotify`) — Original 2017 project being modernized

## Resume Bullet Point (draft)

> Built CortexDJ, a full-stack AI application that classifies EEG brain-wave data into emotional states using a custom dual-head PyTorch CNN (EEGNet), orchestrates session analysis and Spotify playlist curation through a Pydantic AI agent with 9 tools across 4 capabilities, and streams responses via a Next.js chat UI — combining signal processing (bandpass filtering, differential entropy), async FastAPI, and agentic RAG patterns
