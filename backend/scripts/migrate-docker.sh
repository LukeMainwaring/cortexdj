#!/bin/bash

docker compose run --rm backend python -m alembic -c src/cortexdj/alembic.ini upgrade head
