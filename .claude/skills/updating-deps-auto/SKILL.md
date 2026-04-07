---
name: updating-deps-auto
description: "Autonomous dependency update: bump all deps, validate with auto-fix, research changelogs, apply refactors, run code review, and create a PR. Designed for scheduled/unattended execution."
---

# Autonomous Dependency Update

Update all backend and frontend dependencies, refresh library documentation, research changelogs, apply recommended refactors, run code review, and create a PR — all without human interaction.

> This is the autonomous variant of `updating-deps`. For interactive use with review checkpoints, use `/updating-deps` instead.

## Phase 1: Branch Setup

Create a dedicated branch for the update:

```bash
git checkout -b update-deps/$(date +%Y-%m-%d)
```

If the branch already exists (re-running same day), check it out instead:

```bash
git checkout update-deps/$(date +%Y-%m-%d)
```

## Phase 2: Discover Outdated Versions

### Backend

1. Read `backend/pyproject.toml` and note all current dependency version floors (the `>=X.Y.Z` values)
2. Resolve latest versions:

```bash
uv lock --upgrade --directory backend
```

3. Read `backend/uv.lock` to find the resolved version for each dependency listed in `pyproject.toml`

### Frontend

1. Check outdated packages:

```bash
pnpm -C frontend outdated --json
```

2. Note current vs latest for all packages

### Early Exit Check

If no versions changed in either backend or frontend, output "All dependencies are already up to date" and **stop here** — no branch, commit, or PR.

### Record Summary

Compose a markdown table showing each dependency, its current version floor/range, and the latest available version. Group by backend and frontend. **Save this table for the PR body.** Proceed immediately to Phase 3.

## Phase 3: Bump Versions

### Backend

Edit `backend/pyproject.toml` to update each `>=X.Y.Z` floor to the latest resolved version from the lockfile. Then sync:

```bash
uv sync --directory backend
```

### Frontend

For dependencies using `^` ranges:

```bash
pnpm -C frontend update --latest
```

For exact-pinned dependencies (no `^` prefix — e.g., `react`, `react-dom`, `next`, and any in `devDependencies` pinned exactly like `@biomejs/biome`, `ultracite`), bump individually:

```bash
pnpm -C frontend add <pkg>@latest
pnpm -C frontend add -D <pkg>@latest  # for devDependencies
```

Update everything — no exclusions.

## Phase 4: Re-download Library Documentation

Download fresh copies of the key AI library documentation used by this project:

```bash
curl -o docs/pydantic-ai-llms-full.txt https://ai.pydantic.dev/llms-full.txt
```

```bash
curl -s https://ai-sdk.dev/llms.txt | awk '/^# AI SDK UI$/{if(!found){found=1; printing=1}} /^# AI_APICallError$/{if(printing){printing=0; exit}} printing' > docs/vercel-ai-sdk-ui.txt
```

## Phase 5: Validate + Auto-Fix

Run linting and type checking. If failures occur, attempt automatic fixes before proceeding.

### Backend

```bash
uv run --directory backend pre-commit run --all-files
```

If the above fails, run it **again** — pre-commit hooks often fix issues on the first run and pass on the second.

### Frontend

```bash
pnpm -C frontend format
pnpm -C frontend lint
```

### Failure Handling

If validation still fails after auto-fix attempts, capture the full error output into a `validation_failures` section. **Do NOT stop.** Proceed to Phase 6 — failures will be flagged in the PR.

## Phase 5.5: Commit 1 — Dependency Bumps

Stage and commit the dependency, lockfile, documentation, and any auto-fix formatting changes:

```bash
git add backend/pyproject.toml backend/uv.lock frontend/package.json frontend/pnpm-lock.yaml docs/pydantic-ai-llms-full.txt docs/vercel-ai-sdk-ui.txt
```

Also stage any files modified by auto-fix (check `git status` for additional changed files from Phase 5).

```bash
git commit -m "chore: bump all dependencies to latest versions"
```

## Phase 6: Changelog Research

For every direct dependency that changed version (listed in `pyproject.toml` or `package.json`, not transitive-only), fetch the GitHub releases page to review what changed between the old and new versions. Use WebFetch on `https://github.com/<owner>/<repo>/releases`.

Focus on releases between the old version (from Phase 2) and the new version. Extract:

- New features and APIs
- Breaking changes
- Deprecations
- New recommended patterns or best practices

Skip any library that had no version change.

## Phase 7: Apply Refactors

Cross-reference the changelog findings from Phase 6 with the actual codebase. Use Grep and Read to search for deprecated patterns, old API usage, or code that could benefit from newly available features.

**Apply the refactors directly** — do not just report them. For each change:

1. Identify the deprecated pattern or new API opportunity in the codebase (specific file and line)
2. Make the code change
3. Focus on: deprecated API migrations, new pattern adoption, breaking change fixes
4. **Skip** speculative or optional improvements — only apply changes that are clearly beneficial and low-risk

After applying all refactors, re-run validation to ensure nothing is broken:

```bash
uv run --directory backend pre-commit run --all-files
pnpm -C frontend format
pnpm -C frontend lint
```

If the re-validation fails, attempt auto-fix (run pre-commit again, format again). If a specific refactor causes persistent failures, **revert that refactor** using `git checkout -- <file>` and note it as a recommendation in the PR body instead.

Keep a record of:
- **Applied refactors**: what changed and where (for the PR body)
- **Skipped recommendations**: refactors that were too risky or speculative (for the PR body)

## Phase 7.5: Commit 2 — Refactors

If any refactoring changes were made in Phase 7, stage and commit them:

```bash
git add -A
git commit -m "refactor: adopt new patterns from dependency upgrades"
```

If no refactors were applicable, skip this commit entirely.

## Phase 8: Code Review

Launch the **code-reviewer** agent to review all changes on the branch compared to `main`. The agent is defined in `.claude/agents/code-reviewer.md`.

### Handling the Verdict

- **"Ship it"**: No action needed. Record the summary for the PR body.
- **"Fix and ship"**: Apply the suggested fixes from warnings and nits.
- **"Needs changes"**: Attempt to fix the critical issues. After fixing, re-run the code-reviewer agent once more to verify.

### Commit 3 — Review Fixes

If any fixes were made based on code review feedback:

```bash
git add -A
git commit -m "fix: address code review feedback on dependency update"
```

If no fixes were needed, skip this commit.

Record the code review summary and verdict for inclusion in the PR body.

## Phase 9: Push + Create PR

### Push

```bash
git push -u origin update-deps/$(date +%Y-%m-%d)
```

### Duplicate Guard

Before creating a PR, check if one already exists for this branch:

```bash
gh pr list --head update-deps/$(date +%Y-%m-%d) --json number
```

If a PR already exists, skip creation and return the existing PR URL.

### Create PR

```bash
gh pr create --title "chore: bump all dependencies ($(date +%Y-%m-%d))" --body "$(cat <<'EOF'
## Summary
- Automated dependency update for YYYY-MM-DD
- X backend deps updated, Y frontend deps updated

## Validation Status
<!-- "All checks passed (pre-commit, frontend lint)" OR "**VALIDATION FAILURES DETECTED** — manual attention required:" followed by error output -->

## Code Review
<!-- Summary and verdict from code-reviewer agent -->

## Version Changes
<!-- Markdown table from Phase 2 -->

## Refactors Applied
<!-- List of changes made in Phase 7, with file paths. "None" if no refactors were applicable -->

## Breaking Changes
<!-- From changelog research. "None detected" if clean -->

## New Patterns / APIs Worth Adopting
<!-- Recommendations that were NOT applied, for future manual consideration -->

## Deprecation Warnings
<!-- From changelog research -->

## Test Plan
- [ ] Review validation failures (if any)
- [ ] Verify `docker compose up -d` starts cleanly
- [ ] Smoke test chat UI end-to-end

🤖 Generated autonomously by Claude Code (`updating-deps-auto`)
EOF
)"
```

**Important:** Replace all placeholder comments above with actual content from the previous phases. The template shows the structure — fill in real data.

### Post-Creation

- If validation has unresolved failures, add the `needs-attention` label:

```bash
gh pr edit <number> --add-label needs-attention
```

- If the PR body exceeds ~5000 characters, summarize each section and link to the full changelog URLs instead of inlining all details.

### Return the PR URL.
