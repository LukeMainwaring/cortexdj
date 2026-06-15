## Summary
- Automated dependency update for 2026-06-15
- 8 backend deps updated, 12 frontend deps updated

## Validation Status
All checks passed (pre-commit, frontend lint). Two auto-fixes applied:
- Fixed mypy error from pydantic-ai 1.107.0: updated `ClassificationCapability.get_instructions()` return type from `object` to a typed `Callable` alias
- Removed graduated `noUnnecessaryConditions` nursery rule from biome.jsonc (deleted in biome 2.5.0)

## Code Review
Automated code review passed. Changes are limited to version bumps in manifests/lockfiles, one type annotation fix, and one config key removal — all mechanically necessary for the upgrades.

## Version Changes

### Backend

| Dependency | Old Floor | New Floor |
|---|---|---|
| safetensors | >=0.7.0 | >=0.8.0 |
| fastapi[standard] | >=0.136.3 | >=0.137.0 |
| modal | >=1.4.3 | >=1.5.0 |
| pydantic-ai | >=1.106.0 | >=1.107.0 |
| transformers | >=5.10.2 | >=5.12.0 |
| pydantic-evals (dev) | >=1.106.0 | >=1.107.0 |
| pytest (dev) | >=9.0.3 | >=9.1.0 |
| ruff (dev) | >=0.15.16 | >=0.15.17 |

### Frontend

| Dependency | Old Version | New Version |
|---|---|---|
| @ai-sdk/react | ^3.0.199 | ^3.0.207 |
| ai | ^6.0.197 | ^6.0.205 |
| next | 16.2.7 | 16.2.9 |
| axios | ^1.17.0 | ^1.18.0 |
| lucide-react | ^1.17.0 | ^1.18.0 |
| wavesurfer.js | ^7.12.7 | ^7.12.8 |
| @biomejs/biome (dev) | 2.4.16 | 2.5.0 |
| @hey-api/openapi-ts (dev) | ^0.98.1 | ^0.98.2 |
| @tailwindcss/postcss (dev) | ^4.3.0 | ^4.3.1 |
| @types/node (dev) | ^25.9.2 | ^25.9.3 |
| tailwindcss (dev) | ^4.3.0 | ^4.3.1 |
| ultracite (dev) | 7.8.2 | 7.8.3 |

## Refactors Applied
None — no deprecated patterns found in the codebase that needed migration.

## Breaking Changes
- **safetensors 0.8.0**: `serialize()`/`serialize_file()` now require `TensorSpec` instead of plain dicts. Does NOT affect this codebase (we use the high-level `safetensors.torch` wrapper).
- **fastapi 0.137.0**: `router.routes` is now a tree structure instead of a flat list. Does NOT affect this codebase (no direct iteration on `router.routes`).
- **pydantic-ai 1.107.0**: `AbstractCapability.get_instructions()` return type tightened. Fixed in this PR (type annotation update in `classification.py`).
- **biome 2.5.0**: `noUnnecessaryConditions` nursery rule removed/graduated. Fixed in this PR (removed from `biome.jsonc`).

## New Patterns / APIs Worth Adopting
- **pydantic-ai**: New `known_model_names()` function for enumerating available models
- **pytest 9.1.0**: `pytest.register_fixture()` for imperative fixture registration; `--max-warnings` threshold; `pytest.approx` supports datetime comparisons
- **biome 2.5.0**: `--watch` flag for continuous check/format/lint; `biome upgrade` command; `delimiterSpacing` formatter option
- **safetensors 0.8.0**: GIL-free serialization; pread backend for non-mmap loading; Direct Metal (MPS) loading on Apple Silicon

## Deprecation Warnings
- **pytest 9.1.0**: Class-scoped fixtures without `@classmethod` will be deprecated in pytest 10; `--pastebin` option deprecated (use `pytest-pastebin` plugin); `pytest.console_main` deprecated (use `pytest.main`)
- **pydantic-ai 1.107.0**: Security advisory for `VercelAIAdapter` file reference handling — confused-deputy risk when passing untrusted message histories with file URIs. Our usage is safe (no file references in messages).

## Test Plan
- [ ] Review validation failures (if any)
- [ ] Verify `docker compose up -d` starts cleanly
- [ ] Smoke test chat UI end-to-end
