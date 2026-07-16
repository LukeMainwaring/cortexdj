#!/bin/bash
# Create the integration-test database (idempotent). The integration tier
# refuses to run unless POSTGRES_DB contains "test", because it runs
# `alembic upgrade`/`downgrade` against whatever database the env points at.
set -euo pipefail

docker compose up -d postgres

echo "Waiting for Postgres to become ready..."
until docker compose exec postgres pg_isready -U postgres -q; do
  sleep 1
done

docker compose exec postgres psql -U postgres -tc \
  "SELECT 1 FROM pg_database WHERE datname = 'cortexdj_test'" | grep -q 1 ||
  docker compose exec postgres createdb -U postgres cortexdj_test

echo "cortexdj_test is ready. Run the tier with:"
echo "  POSTGRES_DB=cortexdj_test uv run --directory backend pytest -m integration"
