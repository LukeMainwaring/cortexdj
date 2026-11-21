#!/usr/bin/env python3
"""PostToolUse: ultracite check on the changed frontend file (report-only).

Mirrors the AGENTS.md "run lint after frontend changes" instruction. Does not
modify files -- on lint errors it exits 2 so the agent sees them and fixes
them (optionally via `pnpm -C frontend format`). No-op for non-frontend or
generated files so it's cheap on every edit.

Serves both harnesses: Claude Code sends `tool_input.file_path`; Codex's
`apply_patch` sends the raw patch in `tool_input.command`, so the touched
paths are read from its `*** ... File:` headers (one patch may touch several).
"""

import json
import os
import re
import subprocess
import sys

SKIP = (
    "frontend/api/generated/",
    "frontend/node_modules/",
    "frontend/.next/",
)
EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json", ".css")
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


def lint_one(rel_in_frontend: str, project_dir: str) -> str | None:
    """Run ultracite on one file; return its output if it found issues."""
    proc = subprocess.run(  # noqa: S603
        ["pnpm", "-C", "frontend", "exec", "ultracite", "check", rel_in_frontend],
        capture_output=True,
        text=True,
        cwd=project_dir,
    )
    if proc.returncode != 0:
        return (proc.stdout + "\n" + proc.stderr).strip()
    return None


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    project_dir = data.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    failures: list[str] = []
    for file_path in touched_paths(data.get("tool_input") or {}):
        norm = os.path.abspath(os.path.join(project_dir, file_path)).replace(os.sep, "/")
        if "/frontend/" not in norm or not norm.endswith(EXTS) or not os.path.exists(norm):
            continue
        rel_in_frontend = norm.split("/frontend/", 1)[1]
        if any(("frontend/" + rel_in_frontend).startswith(s) for s in SKIP):
            continue
        out = lint_one(rel_in_frontend, project_dir)
        if out is not None:
            failures.append(f"ultracite found issues in frontend/{rel_in_frontend}:\n\n{out}")

    if failures:
        print(
            "\n\n".join(failures)
            + "\n\nFix them (or run `pnpm -C frontend format` for auto-fixable ones), then continue.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
