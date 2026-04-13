## Summary
- Automated dependency update for 2026-04-13
- 3 backend direct deps updated, 4 frontend direct deps updated (plus transitive churn)

## Validation Status
All checks passed (backend pre-commit: ruff, mypy, format; frontend: ultracite lint + format).

## Code Review
**Verdict: Ship it.** Routine bump PR. Lockfiles match manifests, requested floors are satisfied, no source code touched, no unrelated changes. Notable transitive bump: `rich` 14.3.3 → 15.0.0 (pulled in via logfire + CLI output formatting); pre-commit and lint exercised those paths cleanly.

## Version Changes

### Backend (direct)
| Package | Old | New |
|---|---|---|
| pydantic-ai | 1.79.0 | 1.80.0 |
| pydantic-evals | 1.79.0 | 1.80.0 |
| mypy | 1.20.0 | 1.20.1 |

### Backend (notable transitive)
| Package | Old | New |
|---|---|---|
| anthropic | 0.93.0 | 0.94.0 |
| cohere | 6.0.0 | 6.1.0 |
| jiter | 0.13.0 | 0.14.0 |
| logfire / logfire-api | 4.31.1 | 4.32.0 |
| mistralai | 2.3.1 | 2.3.2 |
| pydantic-ai-slim / pydantic-graph | 1.79.0 | 1.80.0 |
| python-multipart | 0.0.24 | 0.0.26 |
| rich | 14.3.3 | 15.0.0 |
| boto3 / botocore | 1.42.87 | 1.42.88 |

### Frontend (direct)
| Package | Old | New |
|---|---|---|
| @ai-sdk/react | 3.0.158 | 3.0.160 |
| ai | 6.0.156 | 6.0.158 |
| @tanstack/react-query | 5.97.0 | 5.99.0 |
| ultracite | 7.4.4 | 7.5.6 |

## Refactors Applied
None. All updates are minor/patch releases without deprecations affecting code we use, and a `Grep` for the new pydantic-ai 1.80.0 symbols (`CapabilityOrdering`, `OpenAICompaction`, `AnthropicCompaction`, `wraps`, `wrapped_by`) returned no hits in `backend/src` — there is no existing pattern to migrate.

## Breaking Changes
None detected in the changelog research for any direct dependency upgraded here. Pydantic-ai 1.80.0 changelog explicitly lists no breaking changes; `ultracite` 7.5.0 migrated its internal oxlint/oxfmt config to TypeScript but does not affect consumers that don't manage those configs (we don't); mypy/ai/react-query updates are patch-level fanout.

## New Patterns / APIs Worth Adopting
- **pydantic-ai `OpenAICompaction` / `AnthropicCompaction` (1.80.0)** — server-side message compaction. Could potentially replace or supplement `backend/src/cortexdj/agents/history_processor.py`, which currently summarizes large historical tool results client-side to prevent token bloat. Worth evaluating whether offloading compaction to the model provider would simplify the multi-turn token strategy. **Not applied here** — non-trivial refactor, deserves a dedicated PR with eval coverage.
- **pydantic-ai `CapabilityOrdering` (1.80.0)** — explicit `innermost`/`outermost`/`wraps`/`wrapped_by`/`requires` operators for composing capabilities. Our `agents/capabilities/` setup is currently linear and small, so this is a future-proofing nicety rather than something to retrofit today.

## Deprecation Warnings
None observed in the upgraded versions' release notes.

## Test Plan
- [ ] Verify `docker compose up -d` starts cleanly
- [ ] Smoke test chat UI end-to-end (greeting -> message -> tool call -> SessionVisualization render)
- [ ] Spot-check that `analyze_session` agent narration still cites `trajectory_summary` fields

🤖 Generated autonomously by Claude Code (`updating-deps-auto`)
