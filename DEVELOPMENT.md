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
uv sync --directory backend
```

Install pre-commit hooks:

```bash
uv run --directory backend pre-commit install
```

### Spotify app (optional)

Playlist and library tools stay hidden unless Spotify is connected. To enable them:

1. Create an app at https://developer.spotify.com/dashboard
2. Add redirect URI `http://127.0.0.1:8003/api/spotify/callback`
3. Copy the client ID + secret into `.env` (`SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET`)
4. Run the app, then click **Settings → Spotify → Connect** in the UI to authorize

## Pre-commit hooks

```bash
uv run --directory backend pre-commit run --all-files
```

## Tests

```bash
uv run --directory backend pytest              # unit tests (eval suite excluded)
uv run --directory backend pytest -v           # verbose output
uv run --directory backend pytest tests/test_preprocessing.py  # single file
uv run --directory backend pytest -m eval      # real-model brain_agent eval suite (opt-in)
```

The `eval` marker gates tests that call the real OpenAI API via `brain_agent` — the default `pytest` invocation excludes them via `addopts = "-m 'not eval'"`. Use them as a nightly safety net on `main` or manual spot-checks, not on every PR. See `.claude/rules/backend/pydantic-ai.md` and `backend/tests/evals/` for the suite layout.

## Database migrations

```bash
# Create a new migration
./backend/scripts/create-db-revision-docker.sh "<migration_message>"

# Apply all pending migrations
./backend/scripts/migrate-docker.sh

# Roll back one migration
./backend/scripts/downgrade-db-revision-docker.sh
```

## API Client Generation

After modifying backend API endpoints:

```bash
# Ensure backend is running
docker compose up -d

# Regenerate client
pnpm -C frontend generate-client
```

## ML Development

### DEAP Dataset

See [backend/data/DEAP_SETUP.md](backend/data/DEAP_SETUP.md) for download instructions. Place `.dat` files in `backend/data/deap/`.

### Model Training

For demoing CortexDJ, skip training entirely — run `./backend/scripts/download-checkpoints.sh` to pull the shipped checkpoints from GitHub Releases into `backend/data/checkpoints/`. The commands below retrain from scratch on DEAP; they're only needed when modifying the ML pipeline.

Three trainable models live side by side: the quadrant classifier (EEGNet or CBraMod backbone) and the contrastive EEG↔CLAP encoder.

```bash
# Classifier (arousal/valence quadrant prediction) — CBraMod by default
uv run --directory backend train-model                    # full LOSO (50 epochs, 32 folds)
uv run --directory backend train-model --quick            # 10 epochs, 3 folds
uv run --directory backend train-model --model eegnet     # lightweight backend
uv run --directory backend compare-models                 # side-by-side eval

# Contrastive EEG↔CLAP encoder
uv run --directory backend fetch-deap-audio               # one-time: resolve DEAP stimuli → iTunes m4a
uv run --directory backend train-contrastive              # full run
uv run --directory backend train-contrastive --quick      # 5-epoch smoke test
uv run --directory backend tensorboard --logdir backend/data/tensorboard_runs
```

Checkpoints land in `backend/data/checkpoints/` (gitignored). CBraMod is the runtime default; set `EEG_MODEL_BACKEND=eegnet` in `.env` to use the lightweight model. The contrastive checkpoint is loaded lazily on the first retrieval request.

Label binarization strategies are selectable via `--label-split`; defaults to `median_per_subject` for balanced per-fold labels. Run `uv run --directory backend train-model --help` for the full option set.

### GPU Training (Modal, optional)

Skip this if you downloaded the shipped checkpoints. Full LOSO with CBraMod is 12+ h on Apple Silicon; Modal runs it in ~1 h on A10G (~$1-2).

```bash
# One-time setup — auth + DEAP volume seed
modal setup
modal volume create cortexdj-deap
for f in backend/data/deap/s*.dat; do
  modal volume put cortexdj-deap "$f" "/$(basename "$f")"
done

# Training runs (wrap long runs in caffeinate to keep the client connected)
caffeinate -dim modal run backend/scripts/modal_train.py                          # classifier, full LOSO
modal run backend/scripts/modal_train.py --args="--quick"                         # quick test
modal run backend/scripts/modal_train.py --args="--model eegnet"                  # EEGNet instead
modal run backend/scripts/modal_train.py --command compare-models                 # compare both
caffeinate -dim modal run backend/scripts/modal_train.py --command train-contrastive
```

Fold-level progress persists on the `cortexdj-deap` volume, so preempted runs auto-resume at the last completed fold (`--args="--no-resume"` to start clean). Checkpoints download to `backend/data/checkpoints/` when the run completes.

### Autoresearch (autonomous EEGNet iteration)

A [Karpathy-style](https://github.com/karpathy/autoresearch) overnight experiment loop for the quadrant classifier — an AI coding agent modifies a single training file, runs a 15-minute experiment on Modal, reads the resulting `avg_macro_f1`, and decides keep-or-revert. Targets the EEGNet pipeline on a fixed 28/4 DEAP subject split.

```bash
# Run one experiment manually (~30 min end-to-end: ~15 min cold-start + 15 min training)
uv run --directory backend python scripts/run_autoresearch.py
# or equivalently:
modal run backend/scripts/modal_autoresearch.py

# Local dry-run on CPU — no Modal, short budget — for sanity-checking train.py edits
WALL_CLOCK_BUDGET_SECONDS=30 AUTORESEARCH_RUN_DIR=/tmp/ar_local \
  uv run --directory backend python -m autoresearch.train
```

To run the agent loop, point any coding agent (Claude Code, Codex) at `backend/autoresearch/program.md` in a fresh session. The agent reads that file, forms a hypothesis, edits `train.py`, invokes the wrapper, and tails `experiments/experiments.jsonl` to decide keep-or-revert. Overnight (8h) budgets roughly 12–16 experiments.

Results land in `backend/autoresearch/experiments/` (gitignored): one JSONL row per run, a `best.json` pointer to the current champion, and per-run artifacts (`train.py` snapshot, `stdout.log`, `metrics.json`). See `backend/autoresearch/README.md` for the full workflow, the four contracts the agent must preserve, and design rationale.

### Database Seeding

```bash
# Seed the `sessions` and `eeg_segments` tables from DEAP .dat files
uv run --directory backend seed-sessions                          # all 32 DEAP participants
uv run --directory backend seed-sessions --participants 1 2 3     # specific participants
uv run --directory backend seed-sessions --model eegnet           # use EEGNet checkpoint

# Seed the `track_audio_embeddings` pgvector index for EEG↔CLAP retrieval.
# Pool = (user Spotify saved tracks) ∪ (12 genre-seed searches). Requires Spotify OAuth.
uv run --directory backend seed-track-index                       # default pool size
uv run --directory backend seed-track-index --limit 500           # smaller pool for dev
uv run --directory backend seed-track-index --skip-library        # genre seeds only (no OAuth)
```

One-off verification scripts live under `backend/scripts/` with usage in their own docstrings.

## Continuous Integration

Lint, type-check, and frontend-lint run on every PR — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
