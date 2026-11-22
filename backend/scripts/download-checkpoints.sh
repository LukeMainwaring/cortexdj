#!/bin/bash
# Download the contrastive encoder's checkpoint (contrastive_best.pt) from the GitHub Release
# into backend/data/checkpoints/. Skips it if it already exists. The quadrant classifier's
# checkpoint is never downloaded: `uv run --directory backend train-classifier` makes it.
#
# Usage:
#   ./backend/scripts/download-checkpoints.sh
#
# Override the release tag (useful for testing pre-release builds):
#   RELEASE_TAG=v0.2.0 ./backend/scripts/download-checkpoints.sh
#
# Requires the GitHub CLI (`gh`). Install: https://cli.github.com/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
CHECKPOINTS_DIR="$BACKEND_DIR/data/checkpoints"
RELEASE_TAG="${RELEASE_TAG:-v0.1.0}"
REPO="LukeMainwaring/cortexdj"

if ! command -v gh >/dev/null 2>&1; then
  echo "Error: GitHub CLI (gh) is required. Install from https://cli.github.com/" >&2
  exit 1
fi

mkdir -p "$CHECKPOINTS_DIR"

echo "Downloading the $RELEASE_TAG contrastive checkpoint into $CHECKPOINTS_DIR ..."
gh release download "$RELEASE_TAG" \
  --repo "$REPO" \
  --dir "$CHECKPOINTS_DIR" \
  --pattern 'contrastive_best.pt' \
  --skip-existing

echo
echo "Checkpoint ready:"
ls -lh "$CHECKPOINTS_DIR/contrastive_best.pt"
