## Summary
- Automated dependency update for 2026-04-07
- 3 backend deps updated, 2 frontend deps updated

## Validation Status
All checks passed (pre-commit, frontend lint)

## Code Review
**Verdict: Ship it** -- All dependency version bumps are correct, lockfiles are consistent with the declared versions, and no unexpected code changes are present. Minor hardening suggestions on the GitHub Actions workflow noted but non-blocking.

## Version Changes

### Backend

| Dependency | Old Version | New Version |
|------------|-------------|-------------|
| mne | >=1.11.0 | >=1.12.0 |
| uvicorn | >=0.43.0 | >=0.44.0 |
| pytest | >=9.0.2 | >=9.0.3 |

### Frontend

| Dependency | Old Version | New Version |
|------------|-------------|-------------|
| @ai-sdk/react | ^3.0.148 | ^3.0.153 |
| ai | ^6.0.146 | ^6.0.151 |

## Refactors Applied
None -- all updates are minor/patch releases with no deprecated patterns to migrate.

## Breaking Changes
None detected

## New Patterns / APIs Worth Adopting
- **uvicorn 0.44.0**: WebSocket keepalive pings for websockets-sansio (relevant if using WebSocket connections)
- **mne 1.12.0**: Release notes could not be fully loaded; review manually at https://github.com/mne-tools/mne-python/releases/tag/v1.12.0

## Deprecation Warnings
None detected

## Test Plan
- [ ] Review validation failures (if any)
- [ ] Verify `docker compose up -d` starts cleanly
- [ ] Smoke test chat UI end-to-end

Generated autonomously by Claude Code (`updating-deps-auto`)
