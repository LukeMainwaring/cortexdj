# Development

All commands run from the repo root. `uv` takes `--directory backend`; `pnpm` takes `-C frontend`.

## Setup

```bash
uv sync --directory backend
uv run --directory backend pre-commit install

cp .env.sample .env            # set OPENAI_API_KEY and/or ANTHROPIC_API_KEY

docker compose up -d           # PostgreSQL (pgvector) and the backend on :8003
./backend/scripts/migrate-docker.sh
uv run --directory backend seed-sessions   # EEG sessions from backend/data/deap (see below)

pnpm -C frontend install
pnpm -C frontend generate-client
pnpm -C frontend dev           # http://localhost:3003
```

The model checkpoints live in `backend/data/checkpoints/`, which is not in git;
`./backend/scripts/download-checkpoints.sh` fetches them from a GitHub Release (it needs the
`gh` CLI). The backend container sees `backend/data` at `/app/data` through a bind mount.
The first model load in the container downloads the CBraMod backbone from the Hugging Face hub
into the `huggingface_cache` volume. Without a checkpoint the backend still boots: it logs a
warning and the assistant does not offer `get_model_info`.

`seed-sessions` reads the 32 DEAP participant files (`backend/data/deap/s01.dat` to `s32.dat`;
`backend/data/DEAP_SETUP.md` says how to get them), classifies every 4-second segment with the
checkpoint it loads itself (`--model cbramod` by default, ground-truth labels when there is no
checkpoint), and writes one session per participant. It runs on the host against the Postgres
container, takes several minutes for all 32, and appends: run it once on a fresh database.
`--participants 1 2 3` seeds a subset.

The frontend needs Node 26 or newer (`engines.node` in `frontend/package.json`, enforced by
`engineStrict` in `frontend/pnpm-workspace.yaml`); CI runs 26. `packageManager` pins pnpm, and
pnpm switches to that version on its own.

The coding-agent skills that Pydantic AI, Logfire, and FastAPI ship inside their wheels are
symlinked into `.agents/skills/` from `backend/.venv`, which is not in git. A fresh clone has
three dangling links until you run this, and the link path carries the Python minor version,
so re-run it after a Python bump:

```bash
UV_PROJECT_ENVIRONMENT=backend/.venv uvx library-skills --no-tool-skill install -y \
  -s building-pydantic-ai-agents -s logfire-instrumentation -s fastapi
```

## Checks and tests

Tests live under `backend/tests/` in three tiers that mirror the source slices. A tier is a
directory: name the directory to run it.

- **`unit/`**: fast, no database, no provider. A bare `pytest` runs only these.
- **`integration/`**: real Postgres and HTTP (httpx `AsyncClient`), one transaction per test
  rolled back. It refuses to run unless `POSTGRES_DB` contains `test`, because it applies the
  shipped migrations and one test downgrades to base, and it fails rather than skips when the
  database is missing, so a green run means the tests ran.
- **`evals/`**: real-model agent evals. See below; they cost money.

```bash
uv run --directory backend pytest                                  # unit tier
uv run --directory backend pytest tests/unit/core/test_config.py   # one file
uv run --directory backend mypy src tests                          # strict
./backend/scripts/test-integration.sh                              # unit + integration
./backend/scripts/test-integration.sh tests/integration/features/chat -x   # extra args go to pytest
uv run --directory backend pre-commit run --all-files              # ruff, format, mypy, import-linter
```

`test-integration.sh` runs pytest on the host against the `postgres` container: it starts
the container if it is not running, creates the `cortexdj_test` database if it is
missing, and stops the container again on exit only if it started it. If another Postgres
holds port 5433 it says so and stops. Tests never read the repo-root `.env`
(`tests/conftest.py` turns it off and blocks real model requests), so a run behaves the same
on every machine and never spends provider credit.

## Evals

`./backend/scripts/run-evals.sh` manages Postgres like `test-integration.sh`, migrates the
test database, and runs `pytest -s tests/evals` with the provider keys from `.env`. The
shipped cases reach no database-backed tool, so the migrate step is there for the evals you
add; seed what those need in the script after it.

The tier has three uses, and each one runs the same files at a different repeat count:

- **Golden set** (`test_golden_set.py`): hand-written good and bad replies run through the
  judges with no agent. A real test: it fails if a judge grades a reply the wrong way. Run it
  after changing a rubric in `tests/evals/judges.py` or `EVAL_JUDGE_MODEL`. Once; its labels
  are fixed, so a repeat only measures judge noise.
- **Agent cases** (`test_assistant_eval.py`, cases in `assistant_dataset.py`): the registered
  assistant on a real model. A measurement, not a gate: it prints a report and fails only if a
  case could not run or be scored. Each run's tool calls print below the report. The default of
  3 runs per case is for reading while working on a prompt; a pass rate from a handful of runs
  is a rough number.
- **Baseline runs**: the same agent cases at `EVAL_REPEAT=5` or more, run before and after a
  deliberate prompt or model change, and recorded as a dated section in
  `docs/evals/baseline.md`. More runs of the same cases narrow the run-to-run spread, which is
  what a before-and-after on a fixed set compares. They say nothing about questions the set
  does not contain; only more cases do that.

```bash
./backend/scripts/run-evals.sh                                   # everything, each agent case three times
./backend/scripts/run-evals.sh tests/evals/test_golden_set.py    # judges only
EVAL_REPEAT=1 EVAL_CASES=friendly_greeting ./backend/scripts/run-evals.sh tests/evals/test_assistant_eval.py
EVAL_REPEAT=5 ./backend/scripts/run-evals.sh tests/evals/test_assistant_eval.py   # a baseline run
```

`EVAL_REPEAT` is how many times each agent case runs (default 3); `EVAL_CASES` is a
comma-separated list of case names. Judges run on `EVAL_JUDGE_MODEL`, pinned and independent
of `AGENT_MODEL`, and a tier above the agents' default
(`docs/adr/0004-eval-judges-a-tier-above-the-agents.md`). The shipped cases cost cents a run;
every case and repeat you add scales that, the agents more than the judges. With
`LOGFIRE_TOKEN` set, a run is exported to Logfire under environment `eval`.

The same steps run from the Actions tab (`.github/workflows/eval.yml`) once the provider
keys are repository secrets. Do not run eval and integration jobs against the test database
at the same time. See `.agents/rules/backend/pydantic-ai.md` for the agent rules.

## Observability (Logfire)

`setup_observability(app)` in `backend/src/cortexdj/core/observability.py` instruments
FastAPI, SQLAlchemy, and Pydantic AI. Without a token, spans print to the terminal and nothing
leaves the process; `LOGFIRE_CONSOLE=false` silences that. To export, put a write token in
`.env` (`LOGFIRE_TOKEN=...`, from logfire.pydantic.dev → project → Settings → Write tokens) and
recreate the backend (`docker compose up -d backend`; a plain `restart` keeps the old
environment); the startup log says whether traces are exported. ADR 0003 says why there
are no custom spans: the facts a trace should be searchable by ride on the agent run as
metadata, and the instrumentors record the rest.

**Which agent ran this turn, and what did it do?** In Live, filter on
`attributes->>'gen_ai.conversation.id' = '<thread id>'` (the thread id is in the browser's
URL) and open the `invoke_agent <agent>` span: the agent is in its name, `metadata` holds
whatever an instance's `AgentDeps` records, and its children are the model requests and tool
calls, with the request's SQL alongside.

**This eval case failed; show me the run.** Run one case with `LOGFIRE_TOKEN` set:

```bash
EVAL_REPEAT=1 EVAL_CASES=friendly_greeting ./backend/scripts/run-evals.sh tests/evals/test_assistant_eval.py
```

Open Experiments, pick the run (its metadata names the agent model, provider, and reasoning
effort, the judge model, and the git sha), and open the failed case: its trace holds that run's
`invoke_agent` span.

## Database migrations

```bash
./backend/scripts/create-db-revision-docker.sh "<migration_message>"   # autogenerate against the dev DB
./backend/scripts/migrate-docker.sh                                     # upgrade head
./backend/scripts/downgrade-db-revision-docker.sh                       # one step back
```

The template ships one initial migration and edits it in place while the schema is still the
scaffold's, instead of stacking revisions on an example. A database created before such an edit
has `upgrade head` as a no-op and the old columns: reset it with `docker compose down -v` (the dev
database) or by dropping and recreating `cortexdj_test` (the test database). Once a real
instance has data, add revisions instead.

Review an autogenerated revision before applying it (pgvector's `CREATE EXTENSION`, the
HNSW index options, and named constraints are the things it gets wrong). Never downgrade
without a reason.

Shapes the first real revisions tend to need: a fixed set of kinds is a string column with a
non-native `Enum` (`native_enum=False`), so a new kind is a code change, not a type migration;
a NOT NULL column on a populated table is added with a `server_default` to backfill and the
default dropped in the same revision; and the module docstring names the ADR the change
serves and what a downgrade loses.

## API client

```bash
pnpm -C frontend generate-client   # exports the OpenAPI schema from the checked-out backend, no server needed
pnpm -C frontend typecheck         # tsc, including the generated client's consumers
```

The output lands in `frontend/api/generated/` and is never hand-edited (a PreToolUse hook
blocks it). Generation exports OpenAPI with `uv` and its frozen lockfile: no running backend,
database, `.env`, or provider keys; the exporter supplies build-time settings and does not run
startup hooks. Health and the streaming chat route are filtered out: chat goes through the AI
SDK transport. The generated REST client initializes through `frontend/api/client.ts`, so
consumers do not need side-effect imports.

Regenerate when upgrading `@hey-api/openapi-ts`. `typescript` is held on the TypeScript 6
package (`npm:@typescript/typescript6`), so `typecheck` runs `tsc6`, because openapi-ts uses
a compiler API TypeScript 7 removed. Lift the hold when an openapi-ts release runs on
TypeScript 7 (check the release notes of any version newer than the one locked;
`npm view @hey-api/openapi-ts version` gives the newest): alias `typescript` back to the plain
package, switch `typecheck` to `tsc`, and delete this sentence.

CI regenerates the client and fails on drift. Formatting is separate (`pnpm -C frontend
format`), so generation does not rewrite unrelated frontend files.

## CI and review

`.github/workflows/ci.yml` runs lint, mypy, the unit and integration tiers against a pgvector
Postgres service with a coverage floor (`--cov-fail-under=75`, baseline ~80%; ratchet it up
as coverage grows), the client-generation drift check, and frontend lint and typecheck on
every PR. The eval tier runs manually from the Actions tab (`eval.yml`) with provider secrets.

Review a branch with `/review-changes` (Claude Code) or `$review-changes` (Codex): the
harness's built-in review over `main...HEAD` plus the review checks in [`REVIEW.md`](REVIEW.md), with
an optional second opinion from the other CLI (`solo` skips that; `cross-model` skips the
question). Pass a spec path to also check the diff against what was asked for. For
uncommitted work, use the built-ins directly: `/code-review` or `codex review --uncommitted`.
