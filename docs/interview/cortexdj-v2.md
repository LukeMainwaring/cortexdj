<!-- CortexDJ — v2 (Standard). Drop-in replacement for the prep-doc stub under
     "## Side Projects and AI tooling". Voice: matches the register of the
     completed TLM/Codex deep dives — full-sentence Pitch/Problem, crisp bullets.
     This is the default to paste straight into the prep doc. -->

### CortexDJ — EEG → music, full-stack neural viz

**What it is.** A full-stack app that classifies EEG into emotion quadrants (relaxed / calm / excited / stressed) for every 4-second window, then has an agent curate Spotify playlists grounded in what the brain actually did while listening — not what the user remembers or a recommender guesses. Built on the public DEAP dataset, with an agentic chat UI on top.

**Why this is the story for Precision.** It's the single project that hits Motti's data-viz interest *and* the signal-processing world in the same breath: real neural-signal preprocessing flowing into a live, interactive UI. Per Jeremy's coaching, I lead with the visualization, not the backend.

**Stack.** Next.js 16 with a custom SVG / `motion` trajectory chart · FastAPI + Pydantic AI agent · PyTorch (lightweight EEGNet + pretrained CBraMod transformer) · pgvector · Modal for GPU training.

---

#### Deep dive: Real-time neural-signal visualization

**Pitch.** A stream of per-4-second EEG predictions is meaningless as a table of numbers. I built a custom-SVG, animated chart that *plays* the brain's path through Russell's affect space — semantically color-coded, frame-accurately scrubbable, and rendered inline in the agent conversation — without reaching for a charting library.

**Problem.** The output of the classifier is a *trajectory*: a sequence of arousal/valence points that tells a story over time. Charting libraries (recharts, Chart.js) give you static axes and a polyline; they don't give you a path that draws itself in, a playhead you can drag frame-by-frame, or quadrant backgrounds that carry meaning. The interesting part of this signal is the motion, and the motion is exactly what off-the-shelf charts throw away.

**Approach**
- Map each 4-second window to a point in the valence × arousal plane (Russell's affect space), with four semantic quadrant backgrounds (relaxed / calm / excited / stressed) so position reads as emotion at a glance.
- The backend computes a smoothed rolling-mean path plus summary stats (dominant quadrant, transition count, path length, dispersion, per-quadrant dwell fractions); the frontend animates that.
- Draw the path with a `motion.path` whose `style={{ pathLength: progress }}` reveals the stroke over a 5-second playback.
- Drive `progress` from a `requestAnimationFrame` loop, holding it in a ref so the per-frame updates stay off React's render path; auto-pause at the end.
- Add play/pause and a scrubber; the playhead is `lerp`'d between adjacent smoothed points by fractional progress, so it glides between 4-second samples instead of snapping to the grid, and segment dots fade in as the path passes them.
- Keep recharts for what it's genuinely good at — the band-power stacked-area chart and the arousal/valence timeline — switched via Radix Tabs.
- Render all of this *inline beneath the agent's tool call*: the message renderer switches on `part.type === "tool-<name>"`, so `analyze_session` auto-mounts the session visualization and `retrieve_tracks_from_brain_state` mounts the ranked-tracks panel. Streaming chat is `@ai-sdk/react`'s `useChat` over a thin SSE proxy route to the Pydantic AI backend, with an OpenAPI-generated client wrapped in TanStack Query hooks that skip retry on 404 (missing session) and 503 (missing checkpoint).

**Impact**
- The session reads at a glance: dominant mood, emotional volatility, and where attention dwelled are obvious without parsing a single number.
- Going custom bought the three things the library couldn't do — animated draw-in, an interpolated scrubbable playhead, and semantically colored quadrants — at near-zero bundle cost.
- Because the viz auto-renders for any agent tool call, visualization is a first-class part of the conversation rather than a separate dashboard the user has to go find.

**Demonstrates**
*data-viz craft · animation & interaction engineering · signal→UI rendering · streaming agent UX · knowing when to drop the library*

**Angle for Precision.** This maps directly onto the JD's "rendering engines for high-frequency physiological data (real-time + offline)" and Motti's stated data-viz emphasis. It's the same core problem — make a noisy physiological signal legible and scrubbable in real time — at a hobbyist scale, and I can speak honestly about where the analogy stops (below).

**Likely follow-ups**
- *Q:* "Why build the chart yourself instead of using a library?" → The two things this signal needs — a path that animates itself in via `pathLength`, and a playhead you can scrub frame-by-frame with smooth interpolation — aren't expressible in recharts/Chart.js. Add semantic quadrant backgrounds and a near-zero bundle, and custom SVG wins. I still used recharts where it actually fit (band-power, timeline).
- *Q:* "How do you keep scrubbing smooth between sampled points?" → The playhead isn't snapped to the 4-second grid; its position is `lerp`'d between the two adjacent smoothed points by the fractional part of progress, so it moves continuously even though samples are 4 seconds apart.
- *Q:* "Would this scale to kHz-rate neural data?" → Honestly, not as-is — this renders at roughly 0.25 Hz segment cadence, not raw kHz. What transfers is the pipeline instinct: aggregate/decimate server-side, keep animation off the React render path with refs and RAF, and choose SVG vs Canvas vs WebGL by point count. I'd expect Precision's real-time path to be Canvas/WebGL with server-side decimation.

**Brush up**
- SVG path construction and interpolation; `motion`'s `pathLength`; RAF loops and avoiding render thrash with refs.
- Decimation / windowing strategies for high-density signal rendering; SVG vs Canvas vs WebGL tradeoffs at scale.
- The Vercel AI SDK UI contract (`useChat`, tool-call message parts) since the inline viz hangs off it.

---

#### Compact: EEG signal pipeline & inter-subject calibration

**Pitch.** The unglamorous other half: turning raw DEAP EEG into per-window arousal/valence I can actually trust — without fooling myself through subject leakage.

**Approach**
- DEAP: 32 subjects, 32-channel EEG at 128 Hz, sliced into 4-second windows against self-reported arousal/valence.
- Binarize labels at **each subject's own median** rather than a global threshold — this removes per-subject Likert-scale bias and keeps folds balanced. It's the inter-subject calibration / normalization problem a BCI company faces, in miniature.
- Validate **leave-one-subject-out** (32 folds): every score is on a subject the model never saw, so it reflects generalization, not memorization.
- Two interchangeable backends: **EEGNet**, a 25K-parameter differential-entropy CNN as the from-scratch baseline, and **CBraMod**, a transformer pretrained on the TUEG corpus and fine-tuned with dual arousal/valence heads. CBraMod's asymmetric conditional positional encoding accepts variable channel counts, so the same model spans 32-channel DEAP today and a 4-channel Muse 2 headset later.
- LOSO training runs on Modal GPUs. The agent itself is Pydantic AI with the brain context injected into the system prompt each turn and tool-result history compaction to bound token growth — mentioned only in passing, since it's deliberately not the part of the stack Motti wants to dig into.

**Likely follow-ups**
- *Q:* "How do you handle inter-subject variability?" → That's exactly the BCI calibration problem. I normalize labels to each subject's own median and validate leave-one-subject-out, so the reported number is for an unseen brain, not a memorized one.
- *Q:* "Why the pretrained transformer over the CNN?" → EEGNet is the fast, honest from-scratch baseline; the pretrained CBraMod backbone is the stronger model and the realistic path to few-channel consumer hardware.

---

**What worked / what didn't.** The classifier is LOSO-validated and ships both backends — that half works, with the pretrained transformer the stronger of the two. The EEG↔CLAP contrastive retrieval path is wired end-to-end (a joint encoder plus a pgvector HNSW index) but, as documented in the roadmap, it does **not** produce usable retrieval signal at DEAP scale: 4-second windows encode mood, arousal, and attention — not the timbral and harmonic track identity that dominates LAION-CLAP audio embeddings — and within-subject cross-trial evaluation also plateaued at the random baseline, which *rules out* subject transfer as the bottleneck. I keep this in the writeup deliberately: it's the exact SNR / inter-subject-calibration problem a BCI company lives inside, and being able to state a clean negative result with the reason behind it is the point, not a gap to hide.
