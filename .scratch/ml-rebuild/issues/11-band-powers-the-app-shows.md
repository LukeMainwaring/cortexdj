# 11: Band powers the app shows

**Type:** grilling
**Status:** resolved
**Blocked by:** 06

## Question

A stored Segment carries five band powers (delta, theta, alpha, beta, gamma) from a Welch spectrum in `compute_band_powers`, and the session chart in `frontend/features/sessions/session-visualization.tsx` draws all five. The model now reads four bands with different edges (ticket 06), and delta is empty in DEAP's files, so the delta the app shows is filter roll-off. Do the displayed band powers come from the model's four bands and its band-pass path, or stay a separate Welch computation with their own edges? What happens to delta: dropped (a frontend change, and the app contract reopens that far), kept and stored as it is, or stored as zero? Does a re-seed change what existing sessions show?

## Answer

Resolved 2026-11-22.

- **Delta:** dropped for DEAP. It is not computed, not stored, and not in the system prompt's band list. DEAP's 4Hz high-pass removed it, so what the chart drew was filter roll-off.
- **One path:** the displayed band powers come from the model's own path: the four bands and edges of ticket 06 (theta 4 to 8, alpha 8 to 13, beta 13 to 30, gamma 30 to 45Hz), the same band-pass, and the stored value is the variance of the band-passed 4-second window averaged over the 32 channels. The model reads the log of what the chart draws. One `FREQ_BANDS`; the Welch `compute_band_powers` is deleted.
- **Visible effect, accepted:** variance is a band's total power where the Welch mean was power per Hz, so wide bands (beta, gamma) take a bigger share of the stacked chart than today.
- **Frontend:** the chart keeps its five-band order and colors and draws only the bands present in the data, so delta disappears for DEAP and returns unprompted for a source that has it. This is the one place the app contract reopens. `band_powers` is a JSONB `dict[str, float]` end to end, so there is no migration, no schema change and no client regeneration.
- **Re-seed:** `seed-sessions` replaces a participant's DEAP session: it deletes that participant's existing `deap` session (segments and session tracks go by cascade; tracks stay) and inserts the fresh one, so a re-seed leaves 32 sessions. The new session has a new id, and a thread's brain context that pointed at the old one loses it. Turned down: refusing when sessions exist and resetting the database, which also means restoring the track index again.
- **Why delta is not kept for later:** it is a real band (deep sleep, drowsiness) and consumer apps such as Muse's show it, but awake delta is mostly blinks and movement and carries little for arousal and valence. A source that has it adds one entry to the bands; nothing here closes that door.
- **For the spec to settle:** `thread_brain_context.latest_session_id` is a plain string with no foreign key, so after a replace it would point at a session that is gone. Default: the seed clears it for the replaced session.
- **Follows from this:** the system prompt's band line lists four bands with the new edges; `tests/factories.py` and `test_preprocessing.py` lose delta; `docs/DEMO_WORKFLOWS.md` is unaffected in wording.

## Comments
