#!/bin/bash

revision_message=$1

if [ -z "$revision_message" ]; then
  echo "Error: Please provide a migration message"
  echo "Usage: ./scripts/create-db-revision-docker.sh \"Your migration message\""
  exit 1
fi

docker compose run --rm backend python -m alembic -c src/cortexdj/alembic.ini revision --autogenerate -m "$revision_message"

echo "Formatting migration files..."
cd backend && uv run ruff format src/cortexdj/migrations/versions/
uv run ruff check --fix src/cortexdj/migrations/versions/
