<!-- CortexDJ — v3 (Fuller). Drop-in replacement for the prep-doc stub under
     "## Side Projects and AI tooling". Voice: more narrative connective tissue
     and a light rapport hook; still side-project-sized, not a Cleanlab-length
     appendix. Same skeleton and facts as v1/v2 — only the prose is warmer/fuller. -->

### CortexDJ — EEG → music, full-stack neural viz

**What it is.** CortexDJ is a side project I built to scratch a specific itch: music changes your brain state, and your brain state changes what music you want next — so what would it look like to close that loop? It classifies EEG into emotion quadrants (relaxed / calm / excited / stressed) for every 4-second window, and then an agent curates Spotify playlists grounded in what the brain *actually did* while listening, rather than what the listener remembers or a collaborative-filtering model assumes. It runs on the public DEAP dataset behind an agentic chat UI.

**Why this is the story for Precision.** Of everything I've built on my own time, this is the one that lands in two of Precision's worlds at once — Motti's data-visualization focus and the real-time signal-processing domain he came up in — because it's genuinely full-stack: neural-signal preprocessing on one end, a live interactive rendering of that signal on the other. Following Jeremy's steer, I lead with the visualization and treat the backend as supporting cast. (If there's a spare minute and it's natural, this is also the project where I'd happily nerd out about the BCI calibration problem — it's the part I find genuinely hard, not the part I'm performing.)

**Stack.** Next.js 16 with a hand-built SVG / `motion` trajectory chart · FastAPI + a Pydantic AI agent · PyTorch (a lightweight EEGNet baseline and a pretrained CBraMod transformer) · pgvector · Modal for GPU training runs.

---

#### Deep dive: Real-time neural-signal visualization

**Pitch.** The classifier emits a stream of per-4-second arousal/valence predictions, and as a table of numbers that's completely inert — nobody reads a brain by squinting at a column of floats. So I built a custom-SVG, animated chart that *plays* the brain's path through Russell's affect space: semantically color-coded, scrubbable frame-by-frame, and rendered inline inside the agent conversation — and I did it without a charting library, on purpose.

**Problem.** What the model actually produces is a *trajectory* — a sequence of points that only means something as motion over time. Charting libraries are built for the opposite: static axes, a polyline, a tooltip. They don't give you a path that draws itself in as the session plays, a playhead you can grab and drag through time, or quadrant backgrounds that encode meaning by *where* a point sits. The single most interesting property of this signal — how the brain moves through emotional space — is exactly the thing an off-the-shelf chart discards. So the build was less "pick a library" and more "decide what the right primitive is."

**Approach**
- Map each 4-second window to a point in the valence × arousal plane (Russell's affect space) and lay four semantic quadrant backgrounds underneath (relaxed / calm / excited / stressed), so a glance at *position* reads as emotion before any label is parsed.
- Let the backend do the statistics: it computes a smoothed rolling-mean path plus summary metrics — dominant quadrant, transition count, path length, dispersion, per-quadrant dwell fractions — and the frontend's job is purely to make that legible and playable.
- Render the path as a `motion.path` whose `style={{ pathLength: progress }}` reveals the stroke progressively over a 5-second playback, so the session literally draws itself in.
- Drive `progress` from a `requestAnimationFrame` loop and hold it in a ref, deliberately keeping the per-frame churn off React's render path; the loop auto-pauses when it reaches the end.
- Add play/pause and a scrubber, and interpolate the playhead: its position is `lerp`'d between the two adjacent smoothed points by the fractional part of progress, so it glides continuously instead of snapping to the 4-second sample grid, while segment dots fade in as the path sweeps past them.
- Keep recharts for the jobs it's actually good at — the stacked band-power area chart and the arousal/valence timeline — wired up behind Radix Tabs, so "custom where it matters, library where it doesn't" is a deliberate split, not dogma.
- Wire the whole thing into the conversation rather than a separate dashboard: the message renderer switches on `part.type === "tool-<name>"`, so `analyze_session` auto-mounts the session visualization and `retrieve_tracks_from_brain_state` mounts the ranked-tracks panel. The chat surface is `@ai-sdk/react`'s `useChat` streaming over a thin SSE proxy route to the Pydantic AI backend, with an OpenAPI-generated client wrapped in TanStack Query hooks that intentionally skip retry on 404 (the session doesn't exist) and 503 (the checkpoint isn't loaded).

**Impact**
- A session reads at a glance — dominant mood, how volatile the listener was, and where their attention dwelled are all visible without reading a single number, which is the entire point of the project.
- Going custom is what bought the three properties the library physically couldn't deliver — the animated draw-in, the smoothly interpolated scrubbable playhead, and semantically colored quadrants — and it cost almost nothing in bundle size because there's no chart dependency behind it.
- Because the visualization auto-renders off any agent tool call, the viz *is* the conversation; the user never context-switches to a dashboard, which is the interaction model I'd want for a clinical reviewer too.

**Demonstrates**
*data-viz craft · animation & interaction engineering · signal→UI rendering · streaming agent UX · knowing when to drop the library*

**Angle for Precision.** This is about as on-the-nose as a side project gets for the JD line "rendering engines for high-frequency physiological data (real-time + offline)" and Motti's data-viz emphasis: it's the same fundamental problem — take a noisy physiological signal and make it legible and scrubbable in real time — just at hobby scale. I'd rather walk in able to draw the analogy *and* say precisely where it breaks (next section) than oversell it; in a regulated, signal-heavy shop that honesty is the more useful signal anyway.

**Likely follow-ups**
- *Q:* "Why build the chart yourself instead of grabbing a library?" → Because the two properties this signal actually needs — a path that animates itself in via `pathLength`, and a playhead you can scrub frame-by-frame with smooth interpolation between samples — simply aren't expressible in recharts or Chart.js. Add semantic quadrant backgrounds and a near-zero bundle and the call makes itself. The tell that it's judgment and not dogma: I *kept* recharts for the band-power and timeline views, where it was the right tool.
- *Q:* "How do you keep the scrub smooth when samples are 4 seconds apart?" → The playhead is never snapped to the sample grid. I take the two smoothed points bracketing the current progress and `lerp` between them by the fractional part, so the dot moves continuously even though the underlying data is coarse — same trick you'd use to interpolate any low-rate telemetry for a smooth display.
- *Q:* "Would this scale to kHz-rate neural data?" → Honestly, not as it stands — this renders at roughly 0.25 Hz segment cadence, nowhere near raw kHz. What does transfer is the pipeline instinct: aggregate and decimate server-side so the client renders a summary, keep animation off the React render path with refs and RAF, and choose SVG versus Canvas versus WebGL by point count. At Precision's volumes I'd expect the real-time path to be Canvas/WebGL fed by server-side decimation — which is a question I'd genuinely want to ask about your current rendering pipeline.

**Brush up**
- SVG path construction and interpolation; `motion`'s `pathLength` API; RAF animation loops and avoiding render thrash by keeping state in refs.
- Decimation / windowing strategies for high-density signal rendering; the SVG vs Canvas vs WebGL tradeoff curve as point count grows.
- The Vercel AI SDK UI contract (`useChat`, tool-call message parts), since the inline visualization hangs entirely off it.

---

#### Compact: EEG signal pipeline & inter-subject calibration

**Pitch.** The unglamorous half of the project, and the part I actually find hard: turning raw DEAP EEG into per-window arousal/valence predictions I can trust — which mostly means not quietly fooling myself through subject leakage.

**Approach**
- DEAP: 32 subjects, 32-channel EEG at 128 Hz, sliced into 4-second windows against each subject's self-reported arousal/valence.
- Binarize labels at **each subject's own median** rather than a single global threshold. People use a 1–9 Likert scale completely differently, so a global cutoff mostly measures rating style; the per-subject median strips that bias out and keeps the folds balanced. This is the inter-subject calibration / normalization problem a BCI company lives with, just in miniature.
- Validate **leave-one-subject-out** — 32 folds, each scored on a subject the model has never seen. It's the slower, less flattering protocol, and it's the only one that answers "does this generalize to a new brain?"
- Ship two interchangeable backends: **EEGNet**, a ~25K-parameter differential-entropy CNN as the honest from-scratch baseline, and **CBraMod**, a transformer pretrained on the TUEG corpus and fine-tuned with custom dual arousal/valence heads. CBraMod's asymmetric conditional positional encoding takes a variable channel count, so the same architecture spans 32-channel DEAP now and a 4-channel Muse 2 headset later — the bridge from public-dataset research to consumer hardware.
- LOSO runs on Modal GPUs. The agent layer is Pydantic AI, with the active brain context injected into the system prompt every turn and tool-result history compaction to keep token growth bounded — I'd mention this only in passing, because it's deliberately the part of the stack Motti said he's *less* interested in right now.

**Likely follow-ups**
- *Q:* "How do you deal with inter-subject variability?" → That's the heart of it, and it's the same problem Precision faces in BCI calibration. I normalize the labels to each subject's own median and validate leave-one-subject-out, so every number I'd quote is for a brain the model never trained on — not a memorized one.
- *Q:* "Why bother with the pretrained transformer if the CNN works?" → EEGNet is the fast, honest baseline that tells me whether the task is even learnable from scratch. CBraMod is the stronger model *and* the realistic path to few-channel consumer hardware, which is the direction I actually care about.

---

**What worked / what didn't.** I want to be straight about this one, because the honest version is the more interesting story. The classifier half works: it's LOSO-validated and ships both backends, with the pretrained transformer the stronger of the two. The EEG↔CLAP contrastive retrieval half is fully wired end-to-end — a joint EEG/audio encoder plus a pgvector HNSW index — but, as the roadmap documents plainly, it does **not** produce usable retrieval signal at DEAP scale. Four-second EEG windows encode mood, arousal, and attention; they don't encode the timbral and harmonic *track identity* that dominates LAION-CLAP audio embeddings, and within-subject cross-trial evaluation also plateaued at the random baseline — which is the useful part, because it *rules out* subject transfer as the bottleneck and points the finger squarely at signal content and SNR. I keep this front and center on purpose: it's the exact inter-subject-calibration and signal-to-noise problem a BCI company lives inside every day, and walking in able to state a clean negative result *with the reason behind it* is worth more than a side project with no failures in it.
