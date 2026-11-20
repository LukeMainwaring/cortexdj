# 0001: Test tiers are directories, integration runs on the host, and tests fake only at declared seams

**Status:** accepted (2026-10-06)

## Context

Most of this backend's tests are written by agents, so new tests copy existing ones more than
anyone designs them, and a bad habit in one test spreads. How a tier is selected, where the
integration tier runs, and what a test is allowed to fake all need one answer each that a copied
test cannot get wrong.

## Decision

- **A tier is its directory.** `testpaths = ["tests/unit"]`, and `pytest tests/integration` /
  `pytest tests/evals` run the other tiers by name.
- **Integration tests run on the host** against the `postgres` container, through
  `backend/scripts/test-integration.sh`.
- **Tests see a fixed environment.** They never read the repo `.env`, and real model requests
  raise, everywhere except `tests/evals/`, which calls real models on the provider keys
  `run-evals.sh` loads from `.env`.
- **Tests fake only at seams the source declares.** The list is in
  `.agents/rules/backend/code-conventions.md` (Testing).

## Considered options

| Option | Why not |
| --- | --- |
| Tier markers (`addopts = "-m 'not eval and not integration'"`) | A later `-m` *replaces* addopts, so a path passed without `-m` runs zero tests and reports green, and CI needs a "do not remove this flag" comment. A tier that skips when its database is missing reads the same as a pass; the integration tier fails instead. |
| A purpose-built Docker test image | Rebuilds on every run and leaves Postgres up afterwards. The host script starts Postgres only if it is not running and stops it only if it started it. CI is the clean-environment run. |
| `monkeypatch.setattr` on app internals (the title generator where `stream.py` imports it) | Each patch names a private detail: a refactor that keeps behavior breaks the test, and the next test copies the patch. Instead, title generation is a FastAPI dependency (`get_title_generator`) the tests override. |

## Consequences

- **A feature's shared fakes live in its own `tests/integration/features/<name>/conftest.py`,**
  not in the shared `client` fixture. `tests/integration/conftest.py` never imports feature code;
  a double only one test uses stays in that test.
- **Parallel runs (`pytest-xdist`) are not set up.** Per-test rollback isolates data, but the
  session-wide migration and the migration round-trip test assume one database, so each worker
  would need its own database first.
