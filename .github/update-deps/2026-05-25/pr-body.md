## Summary
- Automated dependency update for 2026-05-25
- 8 backend deps updated, 10 frontend deps updated

## Validation Status
All checks passed (pre-commit, frontend lint)

## Code Review
**Verdict: Ship it** — All version floors match lockfile resolutions, no code changes to review, no inconsistencies found. One note: pydantic-ai jumped 6 minor versions (1.96.1 → 1.102.0); run the eval suite in CI before merging to catch any behavioral changes.

## Version Changes

### Backend

| Dependency | Old Floor | New Floor |
|---|---|---|
| braindecode | >=1.5.0 | >=1.5.1 |
| fastapi[standard] | >=0.136.1 | >=0.136.3 |
| modal | >=1.4.2 | >=1.4.3 |
| numpy | >=2.4.4 | >=2.4.6 |
| pydantic-ai | >=1.96.1 | >=1.102.0 |
| sqlalchemy[asyncio,mypy] | >=2.0.49 | >=2.0.50 |
| transformers | >=5.8.1 | >=5.9.0 |
| uvicorn | >=0.47.0 | >=0.48.0 |
| pydantic-evals (dev) | >=1.96.1 | >=1.102.0 |
| ruff (dev) | >=0.15.13 | >=0.15.14 |

### Frontend

| Dependency | Old Version | New Version |
|---|---|---|
| @ai-sdk/react | ^3.0.184 | ^3.0.193 |
| @tanstack/react-query | ^5.100.10 | ^5.100.14 |
| ai | ^6.0.182 | ^6.0.191 |
| date-fns | ^4.1.0 | ^4.3.0 |
| geist | ^1.7.0 | ^1.7.1 |
| motion | ^12.38.0 | ^12.40.0 |
| @hey-api/openapi-ts (dev) | ^0.97.1 | ^0.97.2 |
| @types/node (dev) | ^25.8.0 | ^25.9.1 |
| @types/react (dev) | ^19.2.14 | ^19.2.15 |
| postcss (dev) | ^8.5.14 | ^8.5.15 |

## Refactors Applied
None — no deprecated patterns were found in the codebase.

## Breaking Changes
None detected that affect this codebase. Notable upstream breaking changes:
- **pydantic-ai**: `Agent` constructor params `prepare_tools=`, `prepare_output_tools=`, `event_stream_handler=` deprecated; `stream_responses()` → `stream_response()`. This codebase already uses the capability-based `prepare_tools` method pattern and does not use the deprecated APIs.
- **transformers 5.9.0**: SAM3/EdgeTAM `text_embeds` input format changed; LRU caching removed from vision models. Not relevant — this project uses CBraMod/EEGNet only.

## New Patterns / APIs Worth Adopting
- **pydantic-ai**: `ctx.enqueue()` / `agent_run.enqueue()` for pending message queues; `top_k` model setting support for Anthropic models; Graph API out of beta
- **pydantic-ai**: `MCPToolset` via `fastmcp-slim[client]` (replaces older MCP approaches)

## Deprecation Warnings
- **pydantic-ai**: `pydantic_ai.ext.aci` module deprecated; `StreamedResponse.usage()` method-style deprecated; evaluation class positional construction deprecated
- **date-fns**: Documentation now references Temporal API as a potential replacement for many use cases

## Test Plan
- [ ] Review validation failures (if any)
- [ ] Verify `docker compose up -d` starts cleanly
- [ ] Smoke test chat UI end-to-end
