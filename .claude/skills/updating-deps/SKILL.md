---
name: updating-deps
description: "Update all dependencies to latest versions, re-download library docs, and review changelogs for refactoring opportunities."
---

# Update Dependencies

> For scheduled/autonomous execution, use the `updating-deps-auto` skill instead.

Update all backend and frontend dependencies to their latest versions, refresh library documentation, validate the build, and review changelogs for refactoring opportunities.

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

### Summary

Present a markdown table showing each dependency, its current version floor/range, and the latest available version. Group by backend and frontend. Wait for user acknowledgment before proceeding to Phase 3.

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

## Phase 5: Validate

Run linting and type checking to catch any issues from the version bumps:

### Backend

```bash
uv run --directory backend pre-commit run --all-files
```

### Frontend

```bash
pnpm -C frontend lint
```

Report any failures with full error output. Do NOT auto-fix — the user will decide how to address issues.

## Phase 6: Changelog Research

For every direct dependency that changed version (listed in `pyproject.toml` or `package.json`, not transitive-only), fetch the GitHub releases page to review what changed between the old and new versions. Use WebFetch on `https://github.com/<owner>/<repo>/releases`.

Focus on releases between the old version (from Phase 2) and the new version. Extract:

- New features and APIs
- Breaking changes
- Deprecations
- New recommended patterns or best practices

Skip any library that had no version change.

## Phase 7: Refactoring Report

Cross-reference the changelog findings from Phase 6 with the actual codebase. Use Grep and Read to search for deprecated patterns, old API usage, or code that could benefit from newly available features.

Present a structured report:

### 1. Version Summary

| Package | Old Version | New Version |
|---------|-------------|-------------|
| ... | ... | ... |

### 2. Breaking Changes

List anything that needs immediate attention to keep the app functional.

### 3. New Patterns / APIs Worth Adopting

For each recommendation, include:
- What the new pattern/API is
- Where in the codebase it applies (specific file paths)
- A brief code example showing the before/after

### 4. Deprecation Warnings

Things currently used in the codebase that are now deprecated and should be migrated.

### 5. Recommended Refactors

Specific, actionable suggestions with file paths. Prioritize by impact.

**Important:** Do NOT apply any refactoring changes. This phase is report-only. The user will decide which refactors to pursue separately.

## Phase 8: Commit

Ask the user before committing. If approved:

1. Stage the dependency and documentation files:
   - `backend/pyproject.toml`
   - `backend/uv.lock`
   - `frontend/package.json`
   - `frontend/pnpm-lock.yaml`
   - `docs/pydantic-ai-llms-full.txt`
   - `docs/vercel-ai-sdk-ui.txt`

2. Commit with message: `chore: bump all dependencies to latest versions`

Do NOT automatically run code-reviewer or create-pr — those are manual follow-up steps.
