#!/bin/bash
set -e

# The settings guards (core/config.py) and the production CORS list key off this; set
# before the migrate step so a development password is refused before anything runs.
export ENVIRONMENT=${ENVIRONMENT:-production}

echo "Running database migrations..."
python -m alembic -c src/cortexdj/alembic.ini upgrade head

echo "Starting production server..."
python -m uvicorn cortexdj.app:app --host 0.0.0.0 --port ${UVICORN_PORT:-8003}
