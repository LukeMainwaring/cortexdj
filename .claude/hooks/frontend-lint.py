#!/usr/bin/env python3
"""PostToolUse: ultracite check on the changed frontend file (report-only).

Mirrors the AGENTS.md "run lint after frontend changes" instruction. Does not
modify files -- on lint errors it exits 2 so the agent sees them and fixes
them (optionally via `pnpm -C frontend format`). No-op for non-frontend or
generated files so it's cheap on every edit.
"""

import json
import os
import subprocess
import sys

SKIP = (
    "frontend/api/generated/",
    "frontend/node_modules/",
    "frontend/.next/",
)
EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json", ".css")


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

    if "/frontend/" not in norm or not norm.endswith(EXTS):
        return 0
    rel_in_frontend = norm.split("/frontend/", 1)[1]
    if any(("frontend/" + rel_in_frontend).startswith(s) for s in SKIP):
        return 0

    proc = subprocess.run(  # noqa: S603
        ["pnpm", "-C", "frontend", "exec", "ultracite", "check", rel_in_frontend],
        capture_output=True,
        text=True,
        cwd=project_dir,
    )
    if proc.returncode != 0:
        out = (proc.stdout + "\n" + proc.stderr).strip()
        print(
            f"ultracite found issues in frontend/{rel_in_frontend}. Fix them "
            "(or run `pnpm -C frontend format` for auto-fixable ones), then "
            f"continue.\n\n{out}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
