## Summary
- Automated dependency update for 2026-06-01
- 4 backend deps updated, 6 frontend deps updated

## Validation Status
All checks passed (pre-commit, frontend lint)

## Code Review
**Ship it** — Clean dependency update with no source code changes. All lockfiles consistent with declared version floors. No critical, warning, or nit-level issues found.

## Version Changes

### Backend

| Package | Previous | Updated |
|---------|----------|---------|
| braindecode | >=1.5.1 | >=1.5.2 |
| pydantic-ai | >=1.102.0 | >=1.104.0 |
| pydantic-evals | >=1.102.0 | >=1.104.0 |
| ruff | >=0.15.14 | >=0.15.15 |

### Frontend

| Package | Previous | Updated |
|---------|----------|---------|
| @ai-sdk/react | ^3.0.193 | ^3.0.195 |
| ai | ^6.0.191 | ^6.0.193 |
| @biomejs/biome | 2.4.15 | 2.4.16 |
| ultracite | 7.8.0 | 7.8.1 |
| date-fns | ^4.3.0 | ^4.4.0 |
| lucide-react | ^1.16.0 | ^1.17.0 |

## Refactors Applied
None — no deprecated patterns or new API opportunities detected in the codebase.

## Breaking Changes
None detected

## New Patterns / APIs Worth Adopting
- **pydantic-ai 1.104.0**: Adds Claude Opus 4.8 model support and MCP `list_prompts`/`get_prompt` methods. Consider adopting MCP prompt management if needed in the future.

## Deprecation Warnings
- **date-fns 4.4.0**: CDN scripts deprecated in favor of `@date-fns/cdn` package. Not applicable — we use ESM imports.

## Test Plan
- [ ] Review validation failures (if any)
- [ ] Verify `docker compose up -d` starts cleanly
- [ ] Smoke test chat UI end-to-end
