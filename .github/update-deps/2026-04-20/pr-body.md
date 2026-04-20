## Summary
- Automated dependency update for 2026-04-20
- 6 backend deps updated, 7 frontend deps updated

## Validation Status
All checks passed (pre-commit, frontend lint)

## Code Review
**Verdict: Ship it** -- Clean, well-scoped dependency update. All version floor bumps are consistent between manifests and lockfiles. No unexpected packages added or removed. Two new transitive deps noted: `fastar` (FastAPI's new Rust file-serving layer) and `joserfc` (authlib's new JOSE implementation). All bumps are patch/minor with no major version changes.

## Version Changes

### Backend

| Dependency | Old Floor | New Floor |
|---|---|---|
| fastapi | >=0.135.3 | >=0.136.0 |
| modal | >=1.4.1 | >=1.4.2 |
| pydantic | >=2.13.1 | >=2.13.2 |
| pydantic-ai | >=1.82.0 | >=1.84.1 |
| pydantic-evals | >=1.82.0 | >=1.84.1 |
| ruff | >=0.15.10 | >=0.15.11 |

### Frontend

| Dependency | Old Version | New Version |
|---|---|---|
| @ai-sdk/react | ^3.0.163 | ^3.0.170 |
| @tanstack/react-query | ^5.99.0 | ^5.99.2 |
| ai | ^6.0.161 | ^6.0.168 |
| axios | ^1.15.0 | ^1.15.1 |
| next | 16.2.3 | 16.2.4 |
| typescript | ^6.0.2 | ^6.0.3 |
| ultracite | 7.5.9 | 7.6.0 |

## Refactors Applied
None -- all changes are bug fixes and compatibility updates with no deprecated patterns in the codebase.

## Breaking Changes
None detected

## New Patterns / APIs Worth Adopting
- **pydantic-ai 1.84.x**: Stateful compaction mode for `OpenAICompaction` -- could reduce token usage on long conversations. Not adopted since it requires architectural evaluation.
- **pydantic-ai 1.84.x**: Claude Opus 4.7 model support added -- available if/when the project switches to Anthropic models.

## Deprecation Warnings
None detected across any updated dependencies.

## Test Plan
- [ ] Review validation failures (if any)
- [ ] Verify `docker compose up -d` starts cleanly
- [ ] Smoke test chat UI end-to-end

Generated autonomously by Claude Code (`updating-deps-auto`)
