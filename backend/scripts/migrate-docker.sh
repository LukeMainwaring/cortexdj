#!/bin/bash
# Apply pending migrations to the dev database. Run from the repo root.
set -e

docker compose run --rm backend python -m alembic -c src/cortexdj/alembic.ini upgrade head
