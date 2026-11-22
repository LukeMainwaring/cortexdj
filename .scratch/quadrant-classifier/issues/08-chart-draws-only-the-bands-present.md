# 08: The session chart draws only the bands present

**Spec:** `.scratch/ml-rebuild/spec.md` (What the app shows)

**What to build:** The session chart's stacked band-power area shows a band only when the session's segments carry it. For a re-seeded DEAP session that means theta, alpha, beta and gamma, with no delta series and no delta legend entry; a source that has delta gets it back without a code change. This is the one frontend change in the effort.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [x] The chart keeps its five-band order and colours and draws a series, and a legend entry, only for bands present in the data
- [x] A session whose segments have all five bands looks as it does today
- [x] A session whose segments have four bands shows four, with each band's share computed over the bands present
- [x] No backend change and no client regeneration
- [x] `pnpm -C frontend format`, `lint` and `typecheck` pass
