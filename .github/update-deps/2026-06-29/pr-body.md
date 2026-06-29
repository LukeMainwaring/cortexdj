## Summary
- Automated dependency update for 2026-06-29
- 8 backend deps updated (incl. pydantic-ai 1.107→2.0), 5 dev deps updated
- 20 frontend deps updated (incl. ai 6→7, @ai-sdk/react 3→4)

## Validation Status
All checks passed (ruff lint, ruff format, mypy strict, frontend format, frontend lint, `next build`).

## Code Review
**Verdict: Ship it.** Three major version bumps with zero source code changes — all verified compatible:
- **pydantic-ai 2.0**: mypy strict passes, all agent/tool/capability imports verified, no tool prepare callbacks returning `None`
- **ai 7.0 + @ai-sdk/react 4.0**: `next build` passes with TypeScript, `ChatInit` API still supports `id`, `messages`, `transport`, `onFinish`, `onError`, `experimental_throttle`
- Lockfile cleanup is expected: pydantic-ai 2.0 dropped provider extras (ag-ui, cohere, groq, mistral, xai, etc.), removing ~25 transitive packages. Frontend added 2 new transitive deps (`@ai-sdk/mcp`, `@workflow/serde`), removed 1 (`@opentelemetry/api`)
- **Follow-up note**: `docs/pydantic-ai-llms-full.txt` and `docs/vercel-ai-sdk-ui.txt` may be stale relative to the new major versions — refresh via interactive `/updating-deps`

## Version Changes

### Backend (`pyproject.toml` floors)

| Package | Old | New | Type |
|---------|-----|-----|------|
| alembic | >=1.18.4 | >=1.18.5 | patch |
| fastapi[standard] | >=0.137.0 | >=0.138.1 | minor |
| modal | >=1.5.0 | >=1.5.1 | patch |
| pydantic-ai | >=1.107.0 | >=2.0.0 | **major** |
| scipy | >=1.17.1 | >=1.18.0 | minor |
| sqlalchemy[asyncio,mypy] | >=2.0.50 | >=2.0.51 | patch |
| torch | >=2.12.0 | >=2.12.1 | patch |
| transformers | >=5.12.0 | >=5.12.1 | patch |
| pydantic-evals (dev) | >=1.107.0 | >=2.0.0 | **major** |
| pytest (dev) | >=9.1.0 | >=9.1.1 | patch |
| ruff (dev) | >=0.15.17 | >=0.15.20 | patch |

### Frontend (`package.json`)

| Package | Old | New | Type |
|---------|-----|-----|------|
| @ai-sdk/react | ^3.0.207 | ^4.0.5 | **major** |
| ai | ^6.0.205 | ^7.0.4 | **major** |
| @radix-ui/* (8 packages) | various | various | patch |
| @radix-ui/react-slot | ^1.2.5 | ^1.3.0 | minor |
| @tanstack/react-query | ^5.101.0 | ^5.101.2 | patch |
| axios | ^1.18.0 | ^1.18.1 | patch |
| lucide-react | ^1.18.0 | ^1.22.0 | minor |
| motion | ^12.40.0 | ^12.42.0 | minor |
| radix-ui | ^1.5.0 | ^1.6.0 | minor |
| recharts | ^3.8.1 | ^3.9.0 | minor |
| @biomejs/biome (dev) | 2.5.0 | 2.5.1 | patch |
| @hey-api/openapi-ts (dev) | ^0.98.2 | ^0.99.0 | minor |
| @types/node (dev) | ^25.9.3 | ^26.0.1 | major |
| postcss (dev) | ^8.5.15 | ^8.5.16 | patch |

## Refactors Applied
None. All major version bumps are backward-compatible with the existing codebase.

## Breaking Changes
Three major version bumps were included, all verified compatible:

- **pydantic-ai 2.0**: Capabilities-first architecture. Tool prepare callbacks returning `None` now raise `TypeError` (project has none). Dropped provider extras for ag-ui, cohere, groq, mistral, xai (~27 transitive packages removed). mypy strict passes.
- **ai (Vercel AI SDK) 7.0**: ESM-only, Node.js >=22 required, `system` deprecated for `instructions`, `onFinish` deprecated for `onEnd` (still works). Frontend build passes.
- **@ai-sdk/react 4.0**: ESM-only, MCP tool security defaults changed. `useChat` API remains compatible.

## New Patterns / APIs Worth Adopting
- `onFinish` → `onEnd` in `useChat` (`components/chat.tsx:45`) — deprecated but still functional
- `system` → `instructions` for AI SDK prompts (server-side only, not used in frontend)
- `toUIMessageStream()` standalone helper replaces `toUIMessageStreamResponse()` on result objects
- Pydantic AI v2 Capabilities system for composable agent configuration (project already uses this pattern)

## Deprecation Warnings
- `onFinish` callback in `useChat` deprecated in favor of `onEnd` (AI SDK v7)
- `experimental_throttle` remains the current name in `@ai-sdk/react` v4

## Test Plan
- [ ] Review validation failures (if any)
- [ ] Verify `docker compose up -d` starts cleanly
- [ ] Smoke test chat UI end-to-end
