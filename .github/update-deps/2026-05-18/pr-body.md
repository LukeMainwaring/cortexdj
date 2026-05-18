## Summary
- Automated dependency update for 2026-05-18
- 3 backend deps updated, 3 frontend deps updated

## Validation Status
All checks passed (pre-commit, frontend lint)

## Code Review
**Ship it** — Straightforward dependency updates with no code changes and all validation passing. Lock files are consistent with manifest changes, no packages added or removed.

## Version Changes

### Backend

| Dependency | Previous | Updated |
|---|---|---|
| numpy | >=2.4.4 | >=2.4.5 |
| pydantic-ai | >=1.96.1 | >=1.97.0 |
| pydantic-evals (dev) | >=1.96.1 | >=1.97.0 |

### Frontend

| Dependency | Previous | Updated |
|---|---|---|
| @ai-sdk/react | ^3.0.184 | ^3.0.186 |
| ai | ^6.0.182 | ^6.0.184 |
| @hey-api/openapi-ts (dev) | ^0.97.1 | ^0.97.2 |

## Refactors Applied
None — no deprecated patterns found in the codebase.

## Breaking Changes
None detected

## New Patterns / APIs Worth Adopting

- **pydantic-ai 1.97.0**: `stream_responses()` is deprecated in favor of `stream_response()` (singular). Not currently used in this codebase, but worth noting if adopted in the future.
- **pydantic-ai 1.97.0**: `MCPServer*` and `FastMCPToolset` classes deprecated in favor of new `MCPToolset`. Not currently used.
- **pydantic-ai 1.97.0**: `pydantic_graph.beta` API moved out of beta. Not currently used.
- **@hey-api/openapi-ts 0.97.2**: Now supports `valibot` and `zod` as response transformers; TanStack query plugins gained `mutationKeys` option.

## Deprecation Warnings

- **pydantic-ai 1.97.0**: `Agent.to_a2a()` and bundled `fasta2a` integration deprecated; migrate to `fasta2a.pydantic_ai` package if adopted.
- **pydantic-ai 1.97.0**: Google provider IDs renamed: `google-gla:` → `google:`, `google-vertex:` → `google-cloud:`.

## Test Plan
- [ ] Review validation failures (if any)
- [ ] Verify `docker compose up -d` starts cleanly
- [ ] Smoke test chat UI end-to-end
