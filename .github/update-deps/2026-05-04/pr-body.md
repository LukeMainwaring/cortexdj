## Summary
- Automated dependency update for 2026-05-04
- 4 backend deps updated, 10 frontend deps updated

## Validation Status
All checks passed (pre-commit, frontend lint)

## Code Review
**Verdict: Fix and ship.** Clean dependency update across all four expected files. Version bumps are consistent between manifests and lockfiles, no version conflicts, no unexpected file changes, no security concerns. One issue found: `@hey-api/openapi-ts` 0.97.1 broke the `postProcess: ["biome:format"]` config because biome.jsonc excludes the generated output directory — fixed by removing the redundant post-processor (the `generate-client` script already runs `pnpm format`).

## Version Changes

### Backend

| Package | Old Version | New Version |
|---------|-------------|-------------|
| psycopg[binary,pool] | >=3.3.3 | >=3.3.4 |
| pydantic-ai | >=1.87.0 | >=1.89.1 |
| pydantic-evals | >=1.87.0 | >=1.89.1 |
| transformers | >=5.6.2 | >=5.7.0 |

### Frontend

| Package | Old Version | New Version |
|---------|-------------|-------------|
| @ai-sdk/react | ^3.0.170 | ^3.0.176 |
| @biomejs/biome | 2.4.13 | 2.4.14 |
| @hey-api/openapi-ts | ^0.96.1 | ^0.97.1 |
| @tanstack/react-query | ^5.100.5 | ^5.100.9 |
| ai | ^6.0.168 | ^6.0.174 |
| axios | ^1.15.2 | ^1.16.0 |
| lucide-react | ^1.11.0 | ^1.14.0 |
| postcss | ^8.5.12 | ^8.5.13 |
| ultracite | 7.6.2 | 7.6.3 |
| use-stick-to-bottom | ^1.1.3 | ^1.1.4 |

## Refactors Applied
- **`frontend/openapi-ts.config.ts`**: Removed `postProcess: ["biome:format"]` — openapi-ts 0.97.1 now validates post-processor output, which fails because `biome.jsonc` excludes `api/generated/`. The `generate-client` script already runs `pnpm format` as its final step, making this redundant.

## Breaking Changes
- **@hey-api/openapi-ts 0.97.0**: 15 breaking changes including `runtimeConfigPath` resolution (now relative to output folder), error interceptor API changes, and request/response objects in interceptors may be `undefined`. Our config doesn't use any of these features — the only impact was the biome post-processor fix above.
- **axios 1.16.0**: Fetch adapter now enforces `maxBodyLength`/`maxContentLength`, basic auth credentials in URLs are URL-decoded, `parseProtocol` requires strict colon separator. Our usage is indirect (via generated client with simple config), so no impact expected.

## New Patterns / APIs Worth Adopting
- **pydantic-ai 1.89.1**: `conversation_id` for correlating tool calls across runs; dynamic capabilities via callable elements; `builtin_tools` parameter in `agent.override()`. Consider adopting `conversation_id` for tracing in a future PR.
- **pydantic-ai 1.88.0**: `service_tier` model setting for Anthropic/Gemini/Vertex priority. Could be useful for production cost management.
- **transformers 5.7.0**: New models (Laguna, DEIMv2, SAM3-LiteText). Image loading ~17% faster via torchvision native decode. No direct applicability to current CLAP usage.
- **@hey-api/openapi-ts 0.97.0**: 10-30x performance improvement on larger specs. TanStack Query `setQueryData` options now available.

## Deprecation Warnings
- **pydantic-ai 1.88.0**: `prepare_output_tools` hook introduced as replacement for output tool preparation. Our capabilities only use function tools, so no action needed.
- **axios 1.16.0**: Internal `unescape()` replaced with UTF-8 encoding. No direct usage in our code.

## Test Plan
- [ ] Review validation failures (if any)
- [ ] Verify `docker compose up -d` starts cleanly
- [ ] Smoke test chat UI end-to-end
- [ ] Run `pnpm -C frontend generate-client` to verify openapi-ts works with new config

🤖 Generated autonomously by Claude Code (`updating-deps-auto`)
