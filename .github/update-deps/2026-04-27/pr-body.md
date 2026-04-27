## Summary
- Automated dependency update for 2026-04-27
- 5 backend deps updated, 5 frontend deps updated

## Validation Status
All checks passed (pre-commit, frontend lint)

## Code Review
**Verdict: Ship it.** All version floor bumps resolve to expected versions in lockfiles. Pinning conventions preserved (exact pins for `@biomejs/biome` and `ultracite`, caret ranges elsewhere). No packages added or removed. No code changes.

## Version Changes

### Backend

| Dependency | Old Floor | New Floor |
|---|---|---|
| fastapi[standard] | >=0.136.0 | >=0.136.1 |
| pydantic-ai | >=1.86.0 | >=1.87.0 |
| pydantic-evals | >=1.86.0 | >=1.87.0 |
| transformers | >=5.6.1 | >=5.6.2 |
| ruff | >=0.15.11 | >=0.15.12 |

### Frontend

| Dependency | Old Version | New Version |
|---|---|---|
| @tanstack/react-query | ^5.99.2 | ^5.100.5 |
| lucide-react | ^1.8.0 | ^1.11.0 |
| @biomejs/biome | 2.4.12 | 2.4.13 |
| postcss | ^8.5.10 | ^8.5.12 |
| ultracite | 7.6.1 | 7.6.2 |

## Refactors Applied
None — no deprecated patterns or breaking API changes detected in the codebase.

## Breaking Changes
None detected.

## New Patterns / APIs Worth Adopting
- **pydantic-ai 1.87.0**: New `HandleDeferredToolCalls` capability and `ProcessEventStream` capability. Could be useful if deferred tool execution or custom event stream processing is needed in the future.
- **ruff 0.15.12**: New preview feature `#ruff:file-ignore` for file-wide rule suppression (requires preview mode).
- **Biome 2.4.13**: Several new nursery rules (`noUnnecessaryTemplateExpression`, `useDomNodeTextContent`, `noLoopFunc`, `useRegexpTest`). Consider enabling selectively.

## Deprecation Warnings
- **lucide-react**: `text-select` icon renamed to `square-dashed-text` (not used in this codebase).

## Test Plan
- [ ] Review validation failures (if any)
- [ ] Verify `docker compose up -d` starts cleanly
- [ ] Smoke test chat UI end-to-end

Generated autonomously by Claude Code (`updating-deps-auto`)
