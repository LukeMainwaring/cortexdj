#!/usr/bin/env python3
"""PostToolUse: ruff --fix + ruff format on changed backend Python files.

Mirrors the ruff steps in .pre-commit-config.yaml (same file scope:
backend/(src|tests)/**.py). mypy --strict is deliberately left to
pre-commit / CI -- it's too slow to run inline on every edit. ruff --fix
rewrites the changed file; unfixable violations exit 2 back to the agent.

Serves both harnesses: Claude Code sends `tool_input.file_path`; Codex's
`apply_patch` sends the raw patch in `tool_input.command`, so the touched
paths are read from its `*** ... File:` headers (one patch may touch several).

F401 (unused import) is deliberately report-only here (--unfixable F401):
an agent often adds an import in one Edit and its usage in the next, and
auto-removing the import in between breaks the file silently. Pre-commit
still auto-fixes F401 at commit time, when no edit is in flight.
"""

import json
import os
import re
import subprocess
import sys

# Matches .pre-commit-config.yaml `files: ^backend/(src|tests)/.*.py$`
PATTERN = re.compile(r"/backend/(src|tests)/.*\.py$")
PATCH_HEADER = re.compile(r"^\*\*\* (?:Add|Update) File: (.+?)\s*$|^\*\*\* Move to: (.+?)\s*$", re.M)


def touched_paths(tool_input: dict) -> list[str]:
    """Paths an edit touches, from either harness's payload shape."""
    direct = tool_input.get("file_path")
    if direct:
        return [direct]
    patch = tool_input.get("command")
    if not isinstance(patch, str) or "*** Begin Patch" not in patch:
        return []
    return [a or b for a, b in PATCH_HEADER.findall(patch)]


def lint_one(rel_in_backend: str, repo_root: str) -> str | None:
    """Fix + format one file; return ruff's output if unfixable issues remain."""
    base = ["uv", "run", "--directory", "backend", "ruff"]
    fix = subprocess.run(  # noqa: S603
        [*base, "check", "--fix", "--unfixable", "F401", rel_in_backend],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    subprocess.run(  # noqa: S603
        [*base, "format", rel_in_backend],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    if fix.returncode != 0:
        return (fix.stdout + "\n" + fix.stderr).strip()
    return None


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    # The session's current directory, which Codex's relative patch paths are relative to. It
    # is not necessarily the repo root (a `cd` in a shell call moves it), so ruff runs from the
    # root read off each file's own path instead.
    base_dir = data.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    failures: list[str] = []
    for file_path in touched_paths(data.get("tool_input") or {}):
        norm = os.path.abspath(os.path.join(base_dir, file_path)).replace(os.sep, "/")
        match = PATTERN.search(norm)
        # legacy/ is the parked pre-port source: excluded from every check until it is ported.
        if not match or "/legacy/" in norm or not os.path.exists(norm):
            continue
        repo_root = norm[: match.start()]
        rel_in_backend = norm[match.start() + len("/backend/") :]  # e.g. src/cortexdj/foo.py
        out = lint_one(rel_in_backend, repo_root)
        if out is not None:
            failures.append(f"ruff reported unfixable issues in backend/{rel_in_backend}:\n\n{out}")

    if failures:
        print("\n\n".join(failures) + "\n\nAddress them, then continue.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
