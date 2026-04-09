FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

FROM python:3.13-slim-bookworm AS app

# Install system dependencies for MNE/scipy
RUN apt-get update && apt-get install -y --no-install-recommends libhdf5-dev libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the venv
COPY --from=builder /app/.venv /app/.venv

COPY scripts/ /app/scripts/
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

FROM app AS local

# src/ comes from a bind mount in docker-compose.yml
COPY tests/ /app/tests/

# Run the app
CMD ["./scripts/run_local.sh"]

FROM app AS hosted

# src/ baked in for production (no bind mount available)
COPY src/ /app/src/

# Run the app
CMD ["./scripts/run_hosted.sh"]
