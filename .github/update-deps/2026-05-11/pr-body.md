## Summary
- Automated dependency update for 2026-05-11
- 6 backend deps updated, 14 frontend deps updated

## Validation Status
All checks passed (pre-commit: ruff, mypy, formatting; frontend: ultracite lint + format)

## Code Review
**Verdict: Ship it** — The dependency bumps are well-scoped, lockfiles are clean, and no code changes were required. Advisory notes:
- `mypy>=2.0.0` is a major version bump that enables stricter defaults; all current code passes but future code may surface new errors that were silent under 1.x
- `braindecode>=1.5.0` should be verified with a training run to confirm convergence metrics are comparable
- `pydantic-ai>=1.93.0` floor is aggressive for a fast-moving library; consider `>=1.93.0,<2` if API churn becomes a problem

## Version Changes

### Backend (`pyproject.toml` floors)

| Dependency | Old Floor | New Floor |
|---|---|---|
| braindecode | >=1.4.0 | >=1.5.0 |
| pydantic-ai | >=1.89.1 | >=1.93.0 |
| pydantic | >=2.13.3 | >=2.13.4 |
| transformers | >=5.7.0 | >=5.8.0 |
| mypy | >=1.20.2 | >=2.0.0 |
| pydantic-evals | >=1.87.0 | >=1.93.0 |

### Frontend (`package.json`)

| Dependency | Old | New |
|---|---|---|
| @ai-sdk/react | ^3.0.176 | ^3.0.179 |
| ai | ^6.0.174 | ^6.0.177 |
| next | 16.2.4 | 16.2.6 |
| react | ^19.2.5 | ^19.2.6 |
| react-dom | ^19.2.5 | ^19.2.6 |
| tailwind-merge | ^3.5.0 | ^3.6.0 |
| @tailwindcss/postcss | ^4.2.4 | ^4.3.0 |
| tailwindcss | ^4.2.4 | ^4.3.0 |
| @biomejs/biome | 2.4.14 | 2.4.15 |
| ultracite | 7.6.3 | 7.7.0 |
| @types/node | ^25.6.0 | ^25.6.2 |
| @types/react | ^19.2.14 | ^19.2.14 |
| @types/react-dom | ^19.2.3 | ^19.2.3 |
| postcss | ^8.5.13 | ^8.5.14 |

## Refactors Applied
None — no deprecated patterns found in the codebase. `prepare_tools` usage is correctly scoped to function tools (compatible with pydantic-ai 1.93). No `retries` parameter usage (deprecated in favor of `output_retries`). No Apex usage (removed in transformers 5.8).

## Breaking Changes
- **transformers 5.8.0**: Apex integration removed (no impact — project does not use Apex)
- **mypy 2.0.0**: Stricter defaults enabled by default (no impact — project already uses `strict = true` and all checks pass)
- **next 16.2.6**: Security patch release with fixes for DoS, SSRF, middleware bypass, and cache poisoning vulnerabilities

## New Patterns / APIs Worth Adopting
- **pydantic-ai 1.93**: New `tool_choice` configuration for finer-grained tool selection control; new `OutputToolCallEvent`/`OutputToolResultEvent` for output tool observability
- **pydantic-ai 1.92**: `output_retries` parameter replaces deprecated `retries` (not currently used, but worth noting for future agent config)
- **pydantic-ai 1.90**: `openai_conversation_id` for stateful conversation management (already adopted via `conversation_id=thread_id` in prior update)
- **tailwindcss 4.3.0**: New scrollbar utilities (`scrollbar-thin`, `scrollbar-thumb-*`, `scrollbar-track-*`), `zoom-*` utilities, stacked `@variant` support
- **braindecode 1.5.0**: New models (EMG2QwertyNet, MetaNeuromotorHand, CodeBrain), `BaseConcatDataset.set_target()` for one-call relabeling

## Deprecation Warnings
- **pydantic-ai 1.93**: Function-tool events for failing output tool calls deprecated in favor of new structured events
- **pydantic-ai 1.92**: `retries` parameter deprecated; use `output_retries` instead

## Test Plan
- [ ] Review validation failures (if any)
- [ ] Verify `docker compose up -d` starts cleanly
- [ ] Smoke test chat UI end-to-end
- [ ] Run a training job with braindecode 1.5 to verify convergence metrics

🤖 Generated autonomously by Claude Code (`updating-deps-auto`)
