# 11: Band powers the app shows

**Type:** grilling
**Status:** open
**Blocked by:** 06

## Question

A stored Segment carries five band powers (delta, theta, alpha, beta, gamma) from a Welch spectrum in `compute_band_powers`, and the session chart in `frontend/features/sessions/session-visualization.tsx` draws all five. The model now reads four bands with different edges (ticket 06), and delta is empty in DEAP's files, so the delta the app shows is filter roll-off. Do the displayed band powers come from the model's four bands and its band-pass path, or stay a separate Welch computation with their own edges? What happens to delta: dropped (a frontend change, and the app contract reopens that far), kept and stored as it is, or stored as zero? Does a re-seed change what existing sessions show?

## Comments
