#!/bin/bash
# Run the eval tier: real models, graded, against the isolated test DB. This spends money:
# cents for the shipped cases at the defaults, and every case and repeat you add scales it
# (DEVELOPMENT.md, Evals). Loads the provider keys from the repo-root .env and migrates the
# test DB, so an eval that reaches a DB-backed tool finds the schema in place. Like
# test-integration.sh, it starts the postgres container if it is not already running and
# stops it again on exit only if this script started it.
# Run from the repo root.
#   ./backend/scripts/run-evals.sh                                     # the whole tier
#   ./backend/scripts/run-evals.sh tests/evals/test_golden_set.py      # only the paths you name
#   EVAL_REPEAT=1 EVAL_CASES=friendly_greeting ./backend/scripts/run-evals.sh tests/evals/test_assistant_eval.py
#   EVAL_REPEAT=5 ./backend/scripts/run-evals.sh tests/evals/test_assistant_eval.py   # a baseline run
#   ./backend/scripts/run-evals.sh --collect-only -q                   # flags alone keep the default path
set -e

if [ ! -f .env ]; then
  echo "No .env at the repo root: the evals need the provider keys from it." >&2
  exit 1
fi

started_postgres=false
if [ -z "$(docker compose ps -q --status running postgres)" ]; then
  started_postgres=true
fi

clean_up() {
  # Remove here whatever the seed step below wrote, so the integration tier still finds the
  # test DB empty, e.g.:
  #   docker compose exec -T postgres psql -U postgres -d cortexdj_test -qc "DELETE FROM <table>" >/dev/null 2>&1 || true
  if [ "$started_postgres" = true ]; then
    docker compose stop postgres >/dev/null 2>&1
  fi
}
trap clean_up EXIT

if ! up_output=$(docker compose up -d --wait postgres 2>&1); then
  echo "$up_output" >&2
  echo "Could not start postgres. If the port is already allocated, another Postgres holds it: stop that one first." >&2
  exit 1
fi
# A container that is up but unpublished would send the tests to whatever else owns the host port.
published_port=$(docker compose port postgres 5432 2>/dev/null || true)
if ! [[ "$published_port" =~ :[1-9][0-9]*$ ]]; then
  echo "postgres is running but its port is not published on the host; another Postgres likely holds it." >&2
  exit 1
fi

# Ensure the isolated test DB exists (the image only creates POSTGRES_DB, the dev database).
docker compose exec -T postgres psql -U postgres -tc \
  "SELECT 1 FROM pg_database WHERE datname='cortexdj_test'" | grep -q 1 \
  || docker compose exec -T postgres createdb -U postgres cortexdj_test

# A variable already set in the shell wins over the same name in --env-file, so this points
# every step below at the container this script started (its compose credentials and
# published port) while the provider keys still come from .env.
export POSTGRES_DB=cortexdj_test POSTGRES_HOST=localhost POSTGRES_PORT=5433
export POSTGRES_USER=postgres POSTGRES_PASSWORD=postgres
run_backend() {
  uv run --env-file ../.env --directory backend "$@"
}

run_backend alembic -c src/cortexdj/alembic.ini upgrade head
# Seed what your cases need after the migrate step, e.g.:
#   run_backend python -m cortexdj.features.<name>.seed

paths="tests/evals"
for arg in "$@"; do
  case "$arg" in tests/*) paths="" ;; esac
done
# -s shows the eval reports, which are the point of the run.
# shellcheck disable=SC2086  # $paths may be empty on purpose
run_backend pytest -s $paths "$@"
