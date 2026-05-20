### CortexDJ — EEG → music, full-stack neural viz
**What it is.** A full-stack app that classifies EEG into mood states and has an agent curate Spotify playlists from what the brain actually did, not just recommendations from listening history and similar users. Built on the public DEAP dataset with an agentic chat UI on top.

**Why it earns time with Motti.** The one project that hits the data-viz interest *and* the signal-processing world in the same breath — real neural-signal preprocessing flowing into a live, interactive UI.

**Stack at a glance.** Next.js 16 with a custom SVG / `motion` trajectory chart · FastAPI + Pydantic AI agent · PyTorch (lightweight EEGNet + pretrained CBraMod transformer) · pgvector · Modal for GPU training.
#### Deep dive 1: Frontend / data-viz
**Pitch.** The classifier output is a *trajectory through emotion space*, not a list of numbers. Standard charting libraries flatten exactly the part that matters — the motion. So I built a custom chart that plays the brain's path through Russell's affect space, fully scrubbable, and lives inline in the agent conversation rather than off on a separate dashboard.

**Approach.**
- **Position is emotion.** Each 4-second window becomes a point in a 2D valence × arousal plane, with four semantic quadrants (relaxed / calm / excited / stressed) as the backdrop. One glance reads as mood.
- **Animation carries the story.** The chart plays itself in over a few seconds — the trajectory unfolds instead of dumping all at once. Play/pause and a scrubber let you replay any moment.
- **Glides between samples.** Even though the data is sampled every 4 seconds, the playhead moves continuously between points instead of snapping. The eye reads motion, not discrete steps.
- **Library where it fits, custom where it doesn't.** Kept a standard charting library for the things it's genuinely good at (band-power and the arousal/valence-over-time chart). The custom chart only exists where the library would have been the worse answer.
- **Lives inline with the agent.** The viz auto-mounts beneath the agent's tool call rather than living on a separate dashboard. Visualization becomes part of the conversation, not a destination.

**Angle for Precision.** Maps directly onto the JD line about "rendering engines for high-frequency physiological data (real-time + offline)" and Motti's stated data-viz emphasis. Same core problem — make a noisy physiological signal legible and scrubbable in real time — at hobbyist scale, with honest caveats kept in the Takeaways block.

**Likely follow-ups.**
- *Q:* How do you keep scrubbing smooth between sampled points? → Interpolate between adjacent points as the playhead moves through the gap — same trick as tweening keyframes. The eye reads continuous motion even though the underlying samples are 4 seconds apart.
- *Q:* Would this scale to kHz-rate neural data? → Honestly, not as-is — I'm rendering one point every 4 seconds, not thousands per second. What transfers is the instinct: decimate server-side, pick rendering tech by point count. At Precision's scale I'd assume Canvas or WebGL with aggressive server-side decimation.
- *Q:* Why build the chart yourself instead of using a library? → The two things this signal really needs — the path animating itself in, and a playhead that scrubs smoothly between samples — just aren't expressible in the off-the-shelf charting APIs. Once I sketched it, custom turned out to be the simpler answer than fighting a library.

**Brush up.**
- Reference [this Claude chat](https://claude.ai/share/7feb0be1-2dc0-4e59-a728-48c99f6dd289) for more detail:
- Decimation / windowing strategies for high-density signal rendering; SVG vs Canvas vs WebGL tradeoffs at scale.
- Frontend-backend communication approaches: polling, websockets, SSE, caching layers
- The Vercel AI SDK UI contract (`useChat`, tool-call message parts) since the inline viz hangs off it.
- Different charting/data-viz libraries

#### Deep dive 2: Data input, signal decoding & preprocessing
**Pitch.** EEG has low signal-to-noise and high inter-subject variability, so the realistic challenge isn't a fancier model — it's a *fair pipeline* that doesn't let the model cheat. I normalize labels per subject so the model can't memorize rating habits, validate strictly leave-one-subject-out so every number is on an unseen brain, and reuse the same windowing code at training and inference. The unglamorous half of the project, but the half that decides whether the numbers actually mean anything.

**Approach.**
- **4-second windows.** Each 60-second trial gets sliced into 15 non-overlapping 4-second segments — long enough for frequency-band powers to stabilize, short enough to track moment-to-moment emotional state.
- **Two preprocessing paths, one shared slicer.** Engineered features (differential entropy across 5 frequency bands per channel) feed the small CNN baseline; raw resampled signal feeds the pretrained transformer. The window-slicer is the same code at training and at inference — no train/predict drift.
- **Labels normalized per subject.** Each subject's 1–9 Likert ratings get split at their *own* median, not a global threshold. Without this, the model learns "this person rates everything a 7" rather than what their brain actually did.
- **Leave-one-subject-out validation.** 32 folds — every reported number is on a brain the model has never seen. The honest evaluation choice, not the favorable one.
- **Cache the expensive parts.** Feature extraction and audio embeddings cache to disk with content-keyed hashes so cold builds happen once. Caches are host-portable so Modal workers reuse a cache built locally.

**Angle for Precision.** This is the BCI inter-subject calibration problem in miniature — every brain calibrates differently, and per-subject normalization plus leave-one-out validation are what separate "works in the lab" from "generalizes in the clinic." Motti's signal-processing background will recognize the pattern instantly.

**Likely follow-ups.**
- *Q:* How do you handle inter-subject variability? → Two ways. At the label level, I normalize each subject's ratings to their own median rather than a global threshold — so the model isn't learning rating habits. At evaluation, I validate leave-one-subject-out: 32 folds, every reported number is on a brain the model has never seen.
- *Q:* Why two model paths? → EEGNet is the from-scratch baseline on engineered features — small, fast, interpretable, and the honest reference point. CBraMod is the pretrained transformer on raw signal — heavier, but the realistic path to deployment because it learns its own representation rather than depending on hand-crafted features.
- *Q:* How would this transfer to different sensor geometries — surface arrays with very different channel counts? → The window-slicing and label-calibration logic is sensor-agnostic; what changes is the model. CBraMod's positional encoding accepts variable channel counts, so the same backbone scales across electrode geometries with fine-tuning — a big chunk of why I went with a pretrained model in the first place.

**Brush up.**
- EEG frequency bands (delta / theta / alpha / beta / gamma) and their physiological correlates.
- Differential entropy as an EEG feature; why DE over raw band power.
- Inter-subject normalization techniques in BCI: per-subject z-scoring, baseline subtraction, calibration tasks.
- Cross-validation strategies for biological data: LOSO vs. within-subject vs. k-fold; what each tells you.
- CBraMod basics: TUEG pretraining corpus, the asymmetric conditional positional encoding trick for variable channel counts.

#### Deep dive 3: ML training — Modal, pretrained transfer, auto-research
**Pitch.** ML objective: *mood decoding* — classify each 4-second EEG window on arousal and valence, and have that generalize to a brain the model has never seen. I worked it from two angles — first the honest from-scratch baseline (a small dual-head CNN on hand-engineered features), then a pretrained EEG transformer fine-tuned for the same task. A separate contrastive track explores a joint EEG/audio embedding for retrieval. Modal handles GPU on demand, and an auto-research loop lets an agent run experiments unattended.

**Approach.**
- **From-scratch baseline first.** A small dual-head CNN (EEGNet style) on hand-engineered differential entropy features — fast, interpretable, and the honest reference point. If a from-scratch baseline can't beat the majority-class predictor on macro-F1, something's wrong upstream and no fancier model will fix it.
- **Then a pretrained EEG transformer.** Fine-tune CBraMod (pretrained on a much larger EEG corpus) for the same dual-head task. The standard two-phase recipe — freeze the backbone first while the new heads catch up to its representations, then unfreeze with a much lower LR on the encoder than on the heads so the pretrained features shift gently rather than get wrecked.
- **Contrastive training as a separate experiment.** Reuses the CBraMod backbone but with a different objective: jointly embed EEG windows and music into one space so an EEG session can retrieve nearest-neighbor tracks. The exploration was deliberate even though the result was complicated — more in the Takeaways block.
- **Modal for on-demand GPU.** Nice tool for the side-project shape — one command runs training on a cloud GPU with the dataset already mounted, preemption-safe so long runs resume rather than restart. No idle infrastructure between experiments.
- **Auto-research as the vision.** An agent edits the training script, runs an experiment on Modal, logs the result, and decides keep-or-revert next iteration. The interesting part isn't the plumbing — it's that ML iteration starts becoming a background process humans *review*, not a thing humans *operate*.

**Angle for Precision.** Modal + auto-research is the agentic-AI-tooling pattern the JD calls for, in miniature — agents experiment unattended, humans handle judgment. And the baseline-then-pretrained progression is the realistic recipe for any new neural-signal task: prove the ground truth with a simple model first, then layer in the foundation-model muscle.

**Likely follow-ups.**
- *Q:* How does the auto-research loop actually work? → An agent edits the training script, runs one experiment on Modal, gets back a scalar metric, and appends a row to an append-only JSONL log. The agent reads the log on the next iteration to decide keep-or-revert. The point is that one Modal invocation equals one fully-recorded experiment — the agent never loses track of what it's tried.
- *Q:* Why two-phase fine-tuning instead of end-to-end? → Pretrained encoder weights are valuable and easy to damage with the wrong learning rate. Training the heads first against a frozen encoder lets them catch up to its representations; then unfreezing with a much lower LR on the encoder lets it shift gently rather than chase the heads.
- *Q:* Why CBraMod over other pretrained EEG models? → Two criteria: compact enough for real-time inference, and flexible channel counts so the same backbone can handle DEAP today and a different sensor geometry later with fine-tuning. I surveyed about 15 alternatives — most were either too heavy, fixed-channel, or pretrained on the wrong domain.

**Brush up.**
- Two-phase / progressive unfreezing fine-tuning patterns; discriminative learning rates.
- Contrastive loss math: InfoNCE, CLIP-style symmetric, soft-target / supervised contrastive variants.
- Agentic ML research loops as a pattern — append-only experiment logs, scalar metric contracts.
- EMA-smoothed early stopping; macro-F1 vs accuracy on skewed labels; per-class recall as the fastest collapse detector.
- The from-scratch baseline → pretrained progression as methodology; when to step up to a foundation model.

#### Takeaways
*(TODO — populate next iteration)*

**Meta-lessons to surface.**

**Where I'd be careful not to overclaim.** *(future home for the contrastive-retrieval clean negative result at DEAP scale — kept as a strength, not a gap)
