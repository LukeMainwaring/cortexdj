---
name: updating-deps
description: "Update all dependencies to latest versions, validate the build, and review changelogs for refactoring opportunities."
disable-model-invocation: true
---

# Update Dependencies

Update all backend and frontend dependencies to their latest versions, validate the build, and review changelogs for refactoring opportunities.

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

   It exits 1 whenever anything is outdated; that is the expected result, not a failure.

2. Note current vs latest for all packages

### Unused and undeclared

Find dependencies nothing uses and imports nothing declares:

```bash
uv run --directory backend --with deptry deptry .
pnpm dlx knip --directory frontend --dependencies
```

Both read their ignores from the repo (`[tool.deptry]` in `backend/pyproject.toml`, `frontend/knip.json`), so every finding is meant to be real. Confirm each before listing it: a dependency is used only when a file the app actually reaches imports it, so an import in a file nothing else imports (a dead component) is a real finding, not a false positive. A confirmed false positive goes in that ignore list instead.

### Summary

Present a markdown table showing each dependency, its current version floor/range, and the latest available version. Group by backend and frontend. Below it, list the unused and undeclared findings, each with its fix: remove the package (and any file only it served), move it to the dev group, or declare it. Wait for user acknowledgment before proceeding to Phase 3.

## Phase 3: Bump Versions

### Backend

Edit `backend/pyproject.toml` to update each `>=X.Y.Z` floor to the latest resolved version from the lockfile. Then sync:

```bash
uv sync --directory backend
```

Bump the pre-commit hook revisions too, so CI's pinned `ruff-pre-commit` matches the ruff in `pyproject.toml`:

```bash
uv run --directory backend pre-commit autoupdate
```

### Frontend

```bash
pnpm -C frontend update --latest
```

This bumps every dependency and keeps each range's style (`^` stays `^`, exact stays exact).

It does not touch the `packageManager` pin. Bump that by hand to the newest pnpm in the pinned major (`npm view pnpm@<major> version`), then run `pnpm -C frontend install` so the lockfile's package-manager section matches; CI installs with the pinned version and its frozen install fails if that section is stale. pnpm holds back any version published less than a day ago (`minimumReleaseAge`, default 1440 minutes): `outdated` and `update --latest` stop at the newest version at least a day old, and CI rejects a lockfile entry younger than that. To take a same-day release on purpose, add `name@version` to `minimumReleaseAgeExclude` in `frontend/pnpm-workspace.yaml`.

Update everything — no exclusions.

### Unused and undeclared

Apply the fixes the user acknowledged in Phase 2, with `uv add`/`uv remove` and `pnpm -C frontend add`/`remove`, so they land in the same commit as the bumps.

### Holds

A **hold** is an upper bound or alias kept until a stated lift condition is met. Find every hold whose comment states a lift condition: comments next to the dependency in `backend/pyproject.toml`, and, since `package.json` takes no comments, the frontend holds described in `DEVELOPMENT.md`. Each names how to check its condition; run that check. If it is met, lift the hold, delete its comment, and validate with the rest of the bump. If not, leave it and name it in the Phase 7 version summary as held, with its condition.

## Phase 4: Check Bundled Library Skills

Library docs come from the installed packages, so they track each bump with no download: Pydantic AI, Logfire, and FastAPI ship coding-agent skills in their wheels, and the `ai` npm package ships its full docs in `frontend/node_modules/ai/docs/`.

The three wheel skills are symlinked into `.agents/skills/` from `backend/.venv`. The symlink embeds the Python minor (`lib/python3.X`), so re-run the install after a Python bump or if `ls -la .agents/skills/` shows a dangling link:

```bash
UV_PROJECT_ENVIRONMENT=backend/.venv uvx library-skills --no-tool-skill install -y \
  -s building-pydantic-ai-agents -s logfire-instrumentation -s fastapi
```

### Vendored Matt Pocock skills

The skills copied from `mattpocock/skills` are listed in `.agents/skills/VENDORED.md` with the upstream commit they were last synced to. Check for drift:

```bash
./scripts/check-vendored-skills.sh
```

For each skill under **Changed since pin**, read the upstream commits and apply them to our copy: copy the upstream file when the skill has no local edits, hand-apply the change when `VENDORED.md` marks it `(customized)`. For each skill under **New upstream skills**, ask the user whether to vendor it. On yes, copy the skill directory into `.agents/skills/` and add it to the **Vendored** list; on no, add it to **Not vendored**. Either way it never shows as new again. Finish by setting `Upstream commit:` in `VENDORED.md` to the upstream head the script printed.

## Phase 5: Validate

Run linting, type checking, and tests to catch any issues from the version bumps:

### Backend

```bash
uv run --directory backend pre-commit run --all-files
```

`pre-commit` covers lint, format, and `mypy --strict`, but not behaviour. Run the unit suite too:

```bash
uv run --directory backend pytest
```

Then run the integration tier to match the CI gate. The script runs unit + integration on the host against an isolated `cortexdj_test` database, starting Postgres if needed and stopping it again if it started it:

```bash
./backend/scripts/test-integration.sh
```

### Frontend

```bash
pnpm -C frontend lint
pnpm -C frontend typecheck
```

Regenerate the API client too; it exports the schema from the checked-out backend, so no server is needed. Lint and `next build` never run the generator, so a generator break otherwise stays hidden until someone changes an endpoint:

```bash
pnpm -C frontend generate-client
git diff --exit-code frontend/api/generated
```

A crash means the bump broke the generator; a non-empty diff means the bump changed the backend schema or the generator's output. Report which.

Report any failures with full error output. Do NOT auto-fix — the user will decide how to address issues.

## Phase 6: Changelog Research

For every direct dependency that changed version (listed in `pyproject.toml` or `package.json`, not transitive-only), review what changed between the old and new versions on its GitHub releases page (`https://github.com/<owner>/<repo>/releases`). Skip any library that had no version change.

Hand the reading to two subagents run in parallel, one for the backend packages and one for the frontend, so the release notes stay out of this session's context. Give each its packages with old and new versions, and ask for the items that touch this codebase, with file paths:

- New features and APIs
- Breaking changes
- Deprecations
- New recommended patterns or best practices

What they return is leads; Phase 7 checks them against the code.

## Phase 7: Refactoring Report

Cross-reference the changelog findings from Phase 6 with the actual codebase. Use Grep and Read to search for deprecated patterns, old API usage, or code that could benefit from newly available features.

Reproduce every claimed bug before it goes in the report: build, run the test, or inspect the generated output, and show the evidence. A changelog or subagent saying something is broken is a lead, not a finding.

Size the report to the update. When every bump is patch or minor and nothing from Phase 6 touches the code, report only the version summary, the holds, and one line saying there is nothing to adopt. Otherwise present the full structured report:

### 1. Version Summary

| Package | Old Version | New Version |
|---------|-------------|-------------|
| ... | ... | ... |

Add a **Vendored skills** row: old pin, new pin, which skills changed, which were added.

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

1. Show what the update actually changed, so the approval is against a concrete diff:

```bash
git status --porcelain
git diff --stat
```

2. If `git status --porcelain` is empty, nothing was updated (deps already at latest). Report this and skip the commit — do not create an empty commit.

3. Otherwise stage and commit everything that changed. The run is on its own `update-deps/<date>` branch with no auto-fixes (Phase 5 is report-only), so the only changes present are the dependency, lockfile, and pre-commit hook bumps, plus the unused and undeclared fixes from Phase 3:

```bash
git add -A
git commit -m "chore: bump all dependencies to latest versions"
```

Then push and open the PR the way `AGENTS.md` (Pull requests) describes, and stop. A diff of version floors, lockfiles, and removed packages needs no review; say so.
