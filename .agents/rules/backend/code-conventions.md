---
paths:
  - "backend/**/*.py"
---

# Backend rules

What the backend code can't show you before you write it. Everything it does show is a review
check in `REVIEW.md`. Test paths are relative to `backend/`, source paths to
`backend/src/cortexdj/`.

## Copy targets

- A route test: `tests/integration/features/sessions/test_sessions_api.py`.
- An agent tool against the real database: `tests/integration/features/sessions/test_sessions_tools.py`.
- A `shared/` seam: `shared/session_lookup.py`, its test `tests/integration/features/sessions/test_sessions_seams.py`, and the registration in `api.py`.
- An agent under `FunctionModel`: `tests/unit/agents/test_hooks.py`.
- A pure-function unit test: `tests/unit/features/chat/test_title_generator.py`.
- An eval case: `tests/evals/assistant_dataset.py`; a judge rubric: `tests/evals/judges.py`.
- A new named agent: `agents/title_agent.py` and its `THREAD_TITLE` entry in `AGENT_BUILDERS`
  (`agents/registry.py`); a tool-using agent calls `build_agent` with its capabilities, and a
  domain's own agent keeps its builder in `features/<name>/agent.py`.

## Testing

Fake only at these seams; everything else in `cortexdj` runs for real in a test
(`docs/adr/0001-test-tiers-and-fakes.md`).

| What is replaced | How |
|---|---|
| The LLM | a registered agent: `agent.override(model=FunctionModel(...))`, or `TestModel` when any valid output will do. An agent or judge the test builds itself takes the model at construction: `Agent(FunctionModel(...))`, `replace(rubric.judge(), model=FunctionModel(...))` |
| The database session (unit tier) | `make_deps()` from `tests/factories.py` |
| The database session (integration tier) | the `client` fixture, which sets `app.dependency_overrides` |
| Title generation | `app.dependency_overrides[get_title_generator]`, already set by `tests/integration/features/chat/conftest.py` |
| The EEG model | `override_eeg_model(checkpoint)` from `features/sessions/eeg_model.py`, with `make_quadrant_checkpoint()` from `tests/factories.py` (untrained models, given metrics); `None` stands for no checkpoint |
| The contrastive encoder | `override_contrastive_encoder(ContrastiveEncoder(model=EegCLAPEncoder(backbone=fake), device=...))` from `features/retrieval/encoder.py` for the tool, `app.dependency_overrides[get_contrastive_encoder]` for the route; `None` stands for no checkpoint. A service test passes the encoder as the argument |
| The DEAP files | the service's `deap_dir=` argument, pointed at a small pickle a test writes under `tmp_path`. Never read or write `backend/data/` in a test |
| The Spotify client | `override_spotify_client(fake)` from `features/spotify/client.py`, with a small hand-written fake for the spotipy client; `override_spotify_client(None)` stands for not connected. To test what the getter itself does with and without a login, add or leave out a `spotify_tokens` row (`create_spotify_token`) instead. Never set `SPOTIFY_CLIENT_ID` in a test: a stored token that needs a refresh would then call Spotify |
| A `shared/` seam (track access, session lookup) | its `override_<thing>(fake)` context manager; `None` stands for never registered |
| Settings | `monkeypatch.setenv` or `delenv`, then `get_settings.cache_clear()` before and after. A test of a settings class constructs it directly (`AgentSettings(AGENT_MODEL=...)`) |

## Pydantic models or dataclasses

Use a Pydantic model where data crosses a boundary and needs validating, serializing, or a JSON
schema: API schemas in `schema.py` (on `BaseSchema`), settings, an agent's structured output,
eval case inputs. Use a dataclass for a container your own code builds, and wherever the library
expects one: `AgentDeps`, capabilities (`AbstractCapability` is a dataclass), and pydantic-evals
evaluators.

## SQLAlchemy, not SQLModel

Models use SQLAlchemy 2.0 directly, whatever the `fastapi` skill suggests: the table in
`model.py` and the API shape in `schema.py` stay separate classes, so a column never reaches the
API by accident.

## Migrations

After generating an Alembic migration, pause and ask whether it looks right before running
`migrate-docker.sh`. Run a downgrade only when the user asks for one.

## Dependencies

Add a package with `uv add <package> --directory backend` (`--dev` for tooling), never by
hand-editing `pyproject.toml`: a remembered version is a stale one, and `uv add` resolves the
current compatible release and updates `uv.lock` in one step. Keep the lower-bound constraint it
writes; pin `==` only for a specific reason.
