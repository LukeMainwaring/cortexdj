# 06: Trajectory and card labels counted per track

**Spec:** `.scratch/ml-rebuild/spec.md` (What the app shows, The assistant)

**What to build:** A session's trajectory, its dominant quadrant and its card label are computed over its tracks, not its 600 segments, so segment-to-segment noise inside a track no longer shows up as hundreds of transitions or decides the label. The assistant narrates a session track by track. The trajectory summary keeps its shape, so the API, the generated client and the frontend are untouched; segments stay stored and charted as they are.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [x] Time in each quadrant, transitions, the dominant quadrant, centroid, dispersion and path length are computed over a session's tracks, from the mean scores each session track already stores
- [x] The summary's path is one point per track and the three-segment rolling mean is gone; a transition's time is the start of the track it lands on
- [x] Quadrant changes between segments of one track produce no transition
- [x] The session card's label keeps its cut-offs (90% "throughout", 60% "mostly", otherwise "mixed"), counted over tracks, and the list's quadrant distribution is counted the same way
- [x] The Session Narrative instructions describe a track-by-track path: transitions cite tracks, not timestamps between segments
- [x] No API schema change and no client regeneration; the generated-client check stays clean
- [x] The existing trajectory unit tests and sessions API tests are updated to the track-level behaviour
- [x] `GLOSSARY.md`'s Trajectory entry matches what is built
