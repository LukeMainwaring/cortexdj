#!/bin/bash
set -e

echo "Running database migrations..."
python -m alembic -c src/cortexdj/alembic.ini upgrade head

echo "Starting development server with hot reload..."
python -m uvicorn cortexdj.app:app --host 0.0.0.0 --port ${UVICORN_PORT:-8003} --reload
