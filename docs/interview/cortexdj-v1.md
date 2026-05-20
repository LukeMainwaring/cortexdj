<!-- CortexDJ — v1 (Tight). Drop-in replacement for the prep-doc stub under
     "## Side Projects and AI tooling". Voice: terse, bullet-dense, fastest skim. -->

### CortexDJ — EEG → music, full-stack neural viz

**What it is.** Classifies EEG into emotion quadrants (relaxed / calm / excited / stressed) per 4-second window, then curates Spotify playlists from what the brain actually did. Public DEAP dataset; agentic chat UI.

**Why this is the story for Precision.** The one project that hits Motti's data-viz interest *and* the signal-processing world at once: real neural-signal preprocessing feeding a live, interactive UI. Lead with the viz.

**Stack.** Next.js 16 + custom SVG/`motion` viz · FastAPI + Pydantic AI · PyTorch (EEGNet + pretrained CBraMod) · pgvector · Modal GPU.

---

#### Deep dive: Real-time neural-signal visualization

**Pitch.** A stream of per-4-second EEG predictions is unreadable as numbers. I built a custom-SVG, animated chart that plays the brain's path through Russell's affect space — scrubbable, semantically color-coded, no charting library.

**Problem.** Recharts/Chart.js give you axes, not a draw-in path you can scrub frame-by-frame. The signal is a *trajectory*, not a bar chart.

**Approach**
- Each 4 s window → a point in valence × arousal; four semantic quadrant backgrounds.
- Backend computes a smoothed (rolling-mean) path + summary stats (dominant quadrant, transitions, path length, dispersion, dwell %).
- `motion.path` with `style={{ pathLength: progress }}` draws the path in over a 5 s playback.
- `requestAnimationFrame` loop, progress held in a ref to keep the hot path off React renders; auto-pause at end.
- Play/pause + scrubber; playhead `lerp`'d between sampled points so it glides between 4 s samples instead of snapping; dots fade in as the path passes them.
- Recharts kept for what it's good at — band-power stacked-area + arousal/valence timeline, in Radix Tabs.
- Auto-rendered inline beneath the agent's tool call (`message.tsx` switches on `tool-<name>`); streaming chat via `@ai-sdk/react` over a thin SSE proxy to the Pydantic AI backend; OpenAPI-generated client + TanStack Query hooks with 404/503-aware retry.

**Impact**
- Reads at a glance: dominant mood, volatility, and dwell are visible without parsing numbers.
- Custom SVG = animated draw-in + frame-accurate scrubbing + semantic quadrants, at near-zero bundle cost.
- Same surface auto-renders for any agent tool call — viz is a first-class part of the conversation, not a separate dashboard.

**Demonstrates**
*data-viz craft · animation & interaction engineering · signal→UI rendering · streaming agent UX · knowing when to drop the library*

**Angle for Precision.** Direct line to the JD's "rendering engines for high-frequency physiological data" and Motti's data-viz emphasis: same problem — make a noisy physiological signal legible and scrubbable in real time — at hobby scale. Honest about the scale gap (see follow-ups).

**Likely follow-ups**
- *Q:* "Why custom SVG over a charting library?" → animated `pathLength` draw-in + an interpolated, scrubbable playhead aren't expressible in recharts; plus semantic quadrant control and ~zero bundle. Used recharts where it *did* fit.
- *Q:* "How does scrubbing stay smooth between samples?" → playhead position is `lerp`'d between adjacent smoothed points by fractional progress, not snapped to the 4 s grid.
- *Q:* "Would this scale to kHz neural data?" → honestly, no — this renders at ~0.25 Hz segment cadence. The transferable part is the *pipeline instinct*: decimate/aggregate server-side, keep animation off the React render path (refs + RAF), and pick SVG vs Canvas/WebGL by point count.

**Brush up**
- SVG path interpolation; `motion` `pathLength`; RAF loops and avoiding render thrash with refs.
- Decimation/windowing for high-density signal rendering; SVG vs Canvas vs WebGL tradeoffs at scale.

---

#### Compact: EEG signal pipeline & inter-subject calibration

**Pitch.** The unglamorous half: turning raw DEAP EEG into trustworthy per-window arousal/valence without fooling myself with leakage.

**Approach**
- DEAP: 32 subjects, 32-ch EEG @ 128 Hz, 4 s windows, self-reported arousal/valence.
- Binarize labels at **each subject's own median** — kills per-subject Likert-scale bias, balances folds. This is the inter-subject calibration/normalization problem Precision lives in, in miniature.
- **Leave-one-subject-out CV** (32 folds) — no subject leakage; the honest generalization number.
- Two backends: **EEGNet** (25K-param differential-entropy CNN baseline) vs pretrained **CBraMod** transformer (TUEG-pretrained, dual arousal/valence heads). CBraMod's asymmetric conditional positional encoding handles variable channel counts → 32-ch DEAP now, 4-ch Muse 2 later.
- Modal for GPU LOSO; agent is Pydantic AI with brain-context injected into the prompt per turn + history compaction to bound tokens (kept deliberately light here — not the story Motti wants).

**Likely follow-ups**
- *Q:* "Inter-subject variability?" → exactly the BCI calibration problem: I normalize labels per subject (own median) and validate leave-one-subject-out so the score reflects an unseen brain, not a memorized one.
- *Q:* "Why a transformer over the CNN?" → EEGNet is the fast from-scratch baseline; the pretrained CBraMod backbone is the stronger one and the path to few-channel consumer hardware.

---

**What worked / what didn't.** The classifier is LOSO-validated and ships both backends — that part works. The EEG↔CLAP contrastive retrieval path is wired end-to-end (encoder + pgvector HNSW index) but, per the roadmap, does **not** produce usable retrieval signal at DEAP scale: 4 s windows encode mood/arousal/attention, not the timbral/harmonic identity CLAP audio embeddings key on, and within-subject eval also plateaued at the random baseline — which *rules out* subject transfer as the bottleneck. I keep this in the writeup on purpose: it's the exact SNR/calibration problem a BCI company lives in, and being able to state a negative result and *why* is the signal.
