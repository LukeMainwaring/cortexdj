#!/bin/bash
# Run the backend tests that need Postgres, on the host, against the isolated test DB.
# Starts the postgres container if it is not already running, and stops it again on
# exit only if this script started it, so a run leaves Docker the way it found it.
# Run from the repo root.
#   ./backend/scripts/test-integration.sh                        # unit + integration
#   ./backend/scripts/test-integration.sh -x                     # same, with extra pytest flags
#   ./backend/scripts/test-integration.sh tests/integration/features/chat   # only the paths you name
set -e

started_postgres=false
if [ -z "$(docker compose ps -q --status running postgres)" ]; then
  started_postgres=true
fi

stop_postgres_if_started() {
  if [ "$started_postgres" = true ]; then
    docker compose stop postgres >/dev/null 2>&1
  fi
}
trap stop_postgres_if_started EXIT

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

# Unit + integration unless the caller names test paths; flags alone keep the default paths.
paths="tests/unit tests/integration"
for arg in "$@"; do
  case "$arg" in tests/*) paths="" ;; esac
done
# shellcheck disable=SC2086  # $paths is two words on purpose
POSTGRES_DB=cortexdj_test uv run --directory backend pytest $paths "$@"
