### CortexDJ — EEG → music, full-stack neural viz
**What it is.** A full-stack app that classifies EEG into mood states and has an agent curate Spotify playlists from what the brain actually did, not just recommendations from listening history and similar users. Built on the public DEAP dataset with an agentic chat UI on top.

**Why it earns time with Motti.** The one project that hits the data-viz interest *and* the signal-processing world in the same breath — real neural-signal preprocessing flowing into a live, interactive UI.

**Stack at a glance.** Next.js 16 with a custom SVG / `motion` trajectory chart · FastAPI + Pydantic AI agent · PyTorch (lightweight EEGNet + pretrained CBraMod transformer) · pgvector · Modal for GPU training.
#### Deep dive 1: Frontend / data-viz
**Pitch.** A stream of per-4-second EEG predictions is meaningless as a table of numbers, and recharts/Chart.js throw away exactly the part of the signal that matters — the motion. I built a custom-SVG, animated chart that *plays* the brain's path through Russell's affect space: semantically color-coded, frame-accurately scrubbable, and rendered inline in the agent conversation, without reaching for a charting library.

**Approach.**
- Map each 4-second window to a point in valence × arousal (Russell's affect space) with four semantic quadrant backgrounds (relaxed / calm / excited / stressed) — position reads as emotion at a glance.
- Backend computes a smoothed rolling-mean path + summary stats (dominant quadrant, transition count, path length, dispersion, per-quadrant dwell fractions); the frontend animates that.
- Path drawn with `motion.path` + `style={{ pathLength: progress }}` for the reveal over a ~5-second playback; `progress` driven by a `requestAnimationFrame` loop held in a ref so per-frame updates stay off React's render path; auto-pause at the end.
- Play/pause + scrubber; the playhead is `lerp`'d between adjacent smoothed points by fractional progress so it glides continuously between 4-sec samples instead of snapping to the grid; segment dots fade in as the path passes them.
- Kept recharts for what it's genuinely good at — the band-power stacked-area chart and the arousal/valence timeline — switched via Radix Tabs.
- Renders *inline beneath the agent's tool call*: the message renderer switches on `part.type === "tool-<name>"`, so `analyze_session` auto-mounts the session viz and `retrieve_tracks_from_brain_state` mounts the ranked-tracks panel. Streaming chat is `@ai-sdk/react`'s `useChat` over a thin SSE proxy to the Pydantic AI backend; OpenAPI-generated client wrapped in TanStack Query hooks that skip retry on 404 (missing session) and 503 (missing checkpoint).

**Angle for Precision.** Maps directly onto the JD line about "rendering engines for high-frequency physiological data (real-time + offline)" and Motti's stated data-viz emphasis. Same core problem — make a noisy physiological signal legible and scrubbable in real time — at hobbyist scale, with honest caveats kept in the Takeaways block.

**Likely follow-ups.**
- *Q:* Why build the chart yourself instead of using a library? → Recharts/Chart.js can't do the two things this signal needs: a path that animates in via `pathLength`, and a frame-scrubbable playhead with smooth interpolation. Add semantic quadrant backgrounds + near-zero bundle and custom SVG wins. Still used recharts where it fit (band-power, timeline).
- *Q:* How do you keep scrubbing smooth between sampled points? → The playhead isn't snapped to the 4-second grid; its position is `lerp`'d between the two adjacent smoothed points by the fractional part of progress, so it moves continuously even though samples are 4 seconds apart.
- *Q:* Would this scale to kHz-rate neural data? → Honestly, not as-is — this renders at ~0.25 Hz segment cadence, not raw kHz. What transfers is the pipeline instinct: aggregate/decimate server-side, keep animation off React's render path via refs + RAF, choose SVG vs Canvas vs WebGL by point count. Precision's real-time path is almost certainly Canvas/WebGL with server-side decimation.

**Brush up.**
- SVG path construction and interpolation; `motion`'s `pathLength`; RAF loops and avoiding render thrash with refs.
- Decimation / windowing strategies for high-density signal rendering; SVG vs Canvas vs WebGL tradeoffs at scale.
- The Vercel AI SDK UI contract (`useChat`, tool-call message parts) since the inline viz hangs off it.

#### Deep dive 2: Data input, signal decoding & preprocessing
*(TODO — populate next iteration)*
**Pitch.**

**Approach.**

**Angle for Precision.**

**Likely follow-ups.**

**Brush up.**
#### Deep dive 3: ML training — Modal, pretrained transfer, auto-research
*(TODO — populate next iteration)*
**Pitch.**

**Approach.**

**Angle for Precision.**

**Likely follow-ups.**

**Brush up.**

#### Takeaways
*(TODO — populate next iteration)*

**Meta-lessons to surface.**

**Where I'd be careful not to overclaim.** *(future home for the contrastive-retrieval clean negative result at DEAP scale — kept as a strength, not a gap)
