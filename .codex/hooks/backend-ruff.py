#!/usr/bin/env python3
"""PostToolUse: ruff --fix + ruff format on changed backend Python files.

Mirrors the ruff steps in .pre-commit-config.yaml (same file scope:
backend/(src|tests)/**.py). mypy --strict is deliberately left to
pre-commit / CI -- it's too slow to run inline on every edit. ruff --fix
rewrites the changed file; unfixable violations exit 2 back to the agent.
"""

import json
import os
import re
import subprocess
import sys

# Matches .pre-commit-config.yaml `files: ^backend/(src|tests)/.*.py$`
PATTERN = re.compile(r"/backend/(src|tests)/.*\.py$")


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    file_path = (data.get("tool_input") or {}).get("file_path")
    if not file_path:
        return 0

    project_dir = data.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    norm = os.path.abspath(os.path.join(project_dir, file_path)).replace(os.sep, "/")

    if not PATTERN.search(norm):
        return 0
    rel_in_backend = norm.split("/backend/", 1)[1]  # e.g. src/cortexdj/foo.py

    base = ["uv", "run", "--directory", "backend", "ruff"]
    fix = subprocess.run(  # noqa: S603
        [*base, "check", "--fix", rel_in_backend],
        capture_output=True,
        text=True,
        cwd=project_dir,
    )
    subprocess.run(  # noqa: S603
        [*base, "format", rel_in_backend],
        capture_output=True,
        text=True,
        cwd=project_dir,
    )

    if fix.returncode != 0:
        out = (fix.stdout + "\n" + fix.stderr).strip()
        print(
            f"ruff reported unfixable issues in backend/{rel_in_backend}. "
            f"Address them, then continue.\n\n{out}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
