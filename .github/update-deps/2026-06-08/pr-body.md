## Summary
- Automated dependency update for 2026-06-08
- 5 backend deps updated, 22 frontend deps updated

## Validation Status
All checks passed (pre-commit, frontend lint)

## Code Review
**Verdict: Fix and ship** — All version floor bumps are correct and lockfiles are consistent. One action item: regenerate the frontend API client (`pnpm -C frontend generate-client`) after merge to verify `@hey-api/openapi-ts` 0.98.x produces unchanged output (could not run in CI environment since the backend server is required). The 0.98.0 changelog states "The generated output should be unaffected."

## Version Changes

### Backend

| Dependency | Old Floor | New Floor |
|---|---|---|
| pydantic-ai | >=1.104.0 | >=1.106.0 |
| pydantic-evals | >=1.104.0 | >=1.106.0 |
| transformers | >=5.9.0 | >=5.10.2 |
| uvicorn | >=0.48.0 | >=0.49.0 |
| ruff | >=0.15.15 | >=0.15.16 |

### Frontend

| Package | Old Version | New Version |
|---|---|---|
| @ai-sdk/react | ^3.0.195 | 3.0.199 |
| @radix-ui/react-alert-dialog | ^1.1.15 | 1.1.16 |
| @radix-ui/react-collapsible | ^1.1.12 | 1.1.13 |
| @radix-ui/react-dialog | ^1.1.15 | 1.1.16 |
| @radix-ui/react-dropdown-menu | ^2.1.16 | 2.1.17 |
| @radix-ui/react-separator | ^1.1.8 | 1.1.9 |
| @radix-ui/react-slot | ^1.2.4 | 1.2.5 |
| @radix-ui/react-tabs | ^1.1.13 | 1.1.14 |
| @radix-ui/react-tooltip | ^1.2.8 | 1.2.9 |
| @tanstack/react-query | ^5.100.14 | 5.101.0 |
| ai | ^6.0.193 | 6.0.197 |
| axios | ^1.16.1 | 1.17.0 |
| geist | ^1.7.1 | 1.7.2 |
| next | 16.2.6 | 16.2.7 |
| radix-ui | ^1.4.3 | 1.5.0 |
| react | ^19.2.6 | 19.2.7 |
| react-dom | ^19.2.6 | 19.2.7 |
| use-stick-to-bottom | ^1.1.4 | 1.1.6 |
| @hey-api/openapi-ts | ^0.97.3 | 0.98.1 |
| @types/node | ^25.9.1 | 25.9.2 |
| @types/react | ^19.2.15 | 19.2.17 |
| ultracite | 7.8.1 | 7.8.2 |

## Refactors Applied
None — no deprecated patterns or clearly beneficial new API adoptions found in the codebase.

## Breaking Changes
None detected. Key notes:
- **@hey-api/openapi-ts 0.98.0**: Internal config/plugin API change; "generated output should be unaffected" per changelog. Verify by regenerating client post-merge.
- **transformers 5.10.x**: New Audio Language Model base class (without LM head), but cortexdj uses CLAP (contrastive model) which is unaffected.
- **uvicorn 0.49.0**: ProxyHeadersMiddleware now consumes (rather than ignores) duplicate forwarding headers — cortexdj does not use ProxyHeadersMiddleware.

## New Patterns / APIs Worth Adopting
- **pydantic-ai deferred loading**: Tools, instructions, model settings, and hooks can now be loaded on-demand rather than upfront. Could reduce startup time if agent initialization becomes heavy — not needed yet.
- **axios zstd compression**: New `advertiseZstdAcceptEncoding` option for Node HTTP adapter. Not relevant for browser-side usage.

## Deprecation Warnings
- **@tanstack/react-query**: `isServer` deprecated in favor of `environmentManager.isServer()` — cortexdj does not use `isServer`.

## Test Plan
- [ ] Review validation failures (if any)
- [ ] Verify `docker compose up -d` starts cleanly
- [ ] Smoke test chat UI end-to-end
- [ ] Run `pnpm -C frontend generate-client` and verify no output changes
