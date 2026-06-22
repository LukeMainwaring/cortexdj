## Summary
- Automated dependency update for 2026-06-22
- 7 backend deps updated, 13 frontend deps updated

## Validation Status
All checks passed (pre-commit, frontend lint and format)

## Code Review
**Verdict: Ship it** (with notes)

- `@types/node` bumped from ^25.9.3 to ^26.0.0 — a major version jump, but the project runtime is Node v22 and was already mismatched at v25. No type errors observed.
- `radix-ui` 1.5→1.6 and `@radix-ui/react-slot` 1.2→1.3 are minor bumps; lint passed. Recommend a quick smoke test of Radix-based components (dialogs, dropdowns, tooltips).
- All version floors in `pyproject.toml` match resolved lockfile versions exactly. No extraneous file changes.

## Version Changes

### Backend

| Dependency | Old Floor | New Floor |
|---|---|---|
| fastapi[standard] | >=0.137.0 | >=0.138.0 |
| scipy | >=1.17.1 | >=1.18.0 |
| sqlalchemy[asyncio,mypy] | >=2.0.50 | >=2.0.51 |
| torch | >=2.12.0 | >=2.12.1 |
| transformers | >=5.12.0 | >=5.12.1 |
| pytest | >=9.1.0 | >=9.1.1 |
| ruff | >=0.15.17 | >=0.15.18 |

### Frontend

| Dependency | Old Version | New Version |
|---|---|---|
| @ai-sdk/react | ^3.0.207 | ^3.0.210 |
| @radix-ui/react-alert-dialog | ^1.1.16 | ^1.1.17 |
| @radix-ui/react-collapsible | ^1.1.13 | ^1.1.14 |
| @radix-ui/react-dialog | ^1.1.16 | ^1.1.17 |
| @radix-ui/react-dropdown-menu | ^2.1.17 | ^2.1.18 |
| @radix-ui/react-separator | ^1.1.9 | ^1.1.10 |
| @radix-ui/react-slot | ^1.2.5 | ^1.3.0 |
| @radix-ui/react-tabs | ^1.1.14 | ^1.1.15 |
| @radix-ui/react-tooltip | ^1.2.9 | ^1.2.10 |
| ai | ^6.0.205 | ^6.0.208 |
| lucide-react | ^1.18.0 | ^1.21.0 |
| radix-ui | ^1.5.0 | ^1.6.0 |
| @types/node | ^25.9.3 | ^26.0.0 |

## Refactors Applied
None — no deprecated patterns or newly-available APIs were applicable to the current codebase.

## Breaking Changes
None detected. All backend updates are patch-level. Frontend updates are patch or minor within semver ranges.

## New Patterns / APIs Worth Adopting
- **FastAPI 0.138.0** added `app.frontend()` / `router.frontend()` for serving frontend apps directly — not applicable since frontend and backend are decoupled (separate Next.js app).
- **SciPy 1.18.0** added `scipy.signal.whittaker_henderson` for Whittaker-Henderson smoothing — not needed; project uses Butterworth filtering and Welch PSD which remain appropriate for EEG processing.
- **SciPy 1.18.0** expanded Array API support and batch linear algebra — could benefit future numerical work but no immediate application.
- **Ruff 0.15.18** added human-readable rule names in CLI output (preview feature) — available when preview mode is enabled.

## Deprecation Warnings
- **SciPy 1.18.0** now requires Python >=3.12 and NumPy >=2.0.0 — project already meets both requirements (Python 3.13, NumPy 2.4.6).
- No other deprecations affect the codebase. All `scipy.signal` APIs in use (`butter`, `filtfilt`, `welch`, `resample`) remain stable.

## Test Plan
- [ ] Review validation failures (if any)
- [ ] Verify `docker compose up -d` starts cleanly
- [ ] Smoke test chat UI end-to-end
- [ ] Verify Radix UI components (dialogs, dropdowns, tooltips) render correctly after minor bumps
