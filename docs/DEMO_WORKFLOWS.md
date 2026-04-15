# Demo Workflows

Prompts and workflows that showcase what CortexDJ can do that general-purpose chatbots (ChatGPT, Claude) cannot. Organized by uniqueness and impressiveness.

---

## The Best Single Demo

> "Analyze my most recent EEG session and build me a playlist that matches my brain state."

This one prompt triggers the full system. The agent calls `list_sessions` to find the latest recording, `analyze_session` for per-segment arousal/valence classification and band power breakdowns, then `find_relaxing_tracks` (or the appropriate mood) to query historical EEG data for tracks that triggered similar brain states — and proposes a `build_mood_playlist` with user confirmation before creating it on Spotify.

**What to watch for:**

- Sequential tool call indicators: "Loading EEG sessions" spinner → checkmark, then "Analyzing brain states" spinner → checkmark
- The agent explains brain activity using real neuroscience terminology — alpha waves, beta activity, frequency band distributions
- Playlist proposal includes track list and mood rationale — the agent waits for your "yes" before calling `build_mood_playlist`
- Brain context badge appears in the chat header showing the dominant mood, arousal, and valence
- Expand any tool call to see raw input/output JSON — segment timelines, band power arrays, arousal/valence scores

---

## Things No Other Chatbot Can Do

### Brain-Derived Music Recommendations

> "Find me tracks that made people relaxed"

The agent queries historical EEG classification data — not listening history or popularity metrics — to find tracks that consistently triggered low-arousal, high-valence brain states during real neural recordings. These are recommendations backed by measured brain responses, not collaborative filtering.

**What to watch for:**

- "Finding relaxing tracks" spinner → checkmark
- Results include track title, artist, dominant brain state, and average arousal/valence scores
- Each track's recommendation is grounded in EEG data: "This track triggered relaxed states in X segments with average arousal 0.23"
- No Spotify connection required — this uses the EEG database directly

**Variations:**

- *"What music made my brain the most excited?"*
- *"Find tracks associated with calm, contemplative states"*
- *"Which songs triggered the most stress?"*

### Plain-Language Neuroscience

> "What was my brain doing during Session 04?"

The agent translates raw EEG classification data into an accessible narrative — explaining dominant emotional states, notable transitions, and the underlying neural signatures. It references specific frequency bands (alpha, beta, theta) and explains what they indicate about cognitive and emotional processing.

**What to watch for:**

- "Interpreting brain activity" spinner → checkmark
- Response includes frequency band explanations: "Strong alpha activity (8-14 Hz) indicates relaxed alertness"
- Emotion quadrant mapping: arousal/valence scores mapped to relaxed, calm, excited, or stressed
- State distribution breakdown: "65% relaxed, 20% calm, 10% excited, 5% stressed"
- Educational without being clinical — the agent makes neuroscience approachable

### Cross-Session Brain Pattern Comparison

> "Compare Session 03 and Session 18 — what changed?"

Side-by-side comparison of brain responses across two different listening sessions. Reveals how the same person's brain states differ across different music or different days.

**What to watch for:**

- "Comparing sessions" spinner → checkmark
- Delta values: arousal and valence differences between sessions
- Dominant state comparison: did the participant shift from stressed to relaxed?
- The agent interprets the deltas — "Your brain was significantly more relaxed during Session 18, with a 0.3 drop in arousal"

### Persistent Brain Context

> "Set my brain context to Session 11"

Then in follow-up messages:

> "Build me a playlist"

The agent persists brain state context (dominant mood, arousal, valence) in the conversation thread. All subsequent interactions are informed by this context — playlist recommendations, analysis framing, and the agent's tone all adapt.

**What to watch for:**

- Brain context badge appears in the chat header with pills for mood, arousal, and valence
- Refresh the page — the badge reappears because context is persisted in the thread's database column, not browser state
- Follow-up requests automatically reference the active brain context without the user repeating it

### Brain-State Track Retrieval (Contrastive)

> "Find me new songs that match how I was feeling during Session 07"

Unlike `build_mood_playlist` (which curates from tracks already labeled in the EEG database), this uses a contrastive EEG↔CLAP encoder to embed the session's raw brain signal and search a pgvector index of pre-computed track audio embeddings. The agent returns Spotify tracks the user may have never heard — recommendations grounded in the joint EEG-audio embedding space, not a keyword filter on a quadrant label.

**What to watch for:**

- The agent picks `retrieve_tracks_from_brain_state`, *not* `find_relaxing_tracks` or `build_mood_playlist` — check the tool name in the collapsible indicator
- A `<RetrievedTracksPanel>` renders beneath the tool call: rank number, title, artist, similarity bar (cosine score with an emerald tone ≥ 0.5, amber ≥ 0.25, grey below)
- Click the play button on any row to hear a 30s iTunes preview inline — only one track plays at a time
- Click the external-link button to open the track in Spotify
- The agent narrates the ranking in plain language ("The top result has a 0.61 similarity to your session's neural fingerprint")

**Variations:**

- *"Suggest some music that sounds like my brain state in that last session"*
- *"What tracks would match the vibe of Session 12?"*
- *"Find new songs for this session's mood"*

**Prereqs:** The retrieval index must be populated (`uv run --directory backend seed-track-index`) and a contrastive checkpoint must exist (`uv run --directory backend train-contrastive` or via Modal). If the index is empty the tool returns a `note` with operator instructions, which the agent relays verbatim instead of hallucinating a recovery.

---

## End-to-End Workflows

### Full Session Deep Dive

A 4-step workflow demonstrating the complete EEG analysis pipeline.

**Step 1 — Discover sessions:**

> "Show me my EEG sessions"

- Agent calls `list_sessions` — "Loading EEG sessions" spinner → checkmark
- A `<SessionListPanel>` card grid renders inline beneath the tool call: each card shows a stable label ("Session 01" … "Session 32"), a derived dominant-state caption ("Mostly excited", "Calm & focused throughout", "Mixed, leaning excited"), a stacked quadrant distribution bar (emerald/sky/amber/rose), and per-session track + segment counts
- The agent's natural-language reply references sessions by their **Session NN** label — UUIDs, participant IDs, and recording timestamps are intentionally hidden
- Each session represents one DEAP listening sitting (40 music videos at ~1 minute each); for demo purposes treat the whole grid as your own listening history

**Step 2 — Analyze a session:**

> "Analyze Session 04"

- Agent resolves the **Session 04** label to its underlying UUID from the prior `list_sessions` output and calls `analyze_session` — "Analyzing brain states" spinner → checkmark
- Response includes:
  - High-level session summary (label, duration, dominant state)
  - Summary with dominant state and arousal/valence averages
  - Per-segment timeline with individual arousal/valence scores and band powers
  - Track information with per-track emotion summaries

**Step 3 — Get the narrative:**

> "Explain what my brain was doing — what do the patterns mean?"

- Agent calls `explain_brain_state` — "Interpreting brain activity" spinner → checkmark
- Natural-language interpretation referencing specific frequency bands
- The agent connects neural patterns to emotional experience: "The sustained alpha dominance suggests you were in a state of relaxed enjoyment"

**Step 4 — Act on the insights:**

> "Build me a playlist of tracks that trigger that same relaxed state"

- Agent calls `find_relaxing_tracks` to query EEG data for mood-matched tracks
- Proposes a playlist name and track list — waits for confirmation
- After user confirms, calls `build_mood_playlist` with `user_confirmed=True`
- If Spotify is connected, the playlist is created on Spotify with a link; otherwise saved locally

**What to watch for across the session:** Brain context badge appears after the analysis and persists through all steps. The agent's language becomes increasingly specific as it builds up context from prior tool results.

### Spotify Library Meets Brain Data

A workflow combining Spotify library access with EEG-derived insights. Requires Spotify connection (Settings > Spotify > Connect).

**Step 1 — Check recent listening:**

> "What have I been listening to recently?"

- Agent calls `get_listening_history` — "Fetching listening history" spinner → checkmark
- Results show recently played tracks with timestamps and Spotify metadata

**Step 2 — Cross-reference with brain data:**

> "Do any of those tracks appear in my EEG sessions? What brain states did they trigger?"

- Agent searches the EEG database for matching tracks
- If matches are found, it shows the brain state data alongside the Spotify listening history
- This bridges subjective taste (what you chose to play) with objective neural response (how your brain reacted)

**Step 3 — Enrich an existing playlist:**

> "Show me my Spotify playlists"

- Agent calls `get_my_playlists` — results show playlist names and track counts

> "Add the top relaxing tracks from my brain data to my 'Chill Vibes' playlist"

- Agent calls `find_relaxing_tracks`, then proposes adding specific tracks to the named playlist
- Waits for confirmation, then calls `add_tracks_to_playlist` with `user_confirmed=True`

**What to watch for:** The agent seamlessly combines EEG data tools with Spotify library tools — querying brain states in one step and writing to Spotify in the next.

### Model Introspection

> "Tell me about the EEG model you're using"

- Agent calls `get_model_info` — "Loading model info" spinner → checkmark
- Response reveals the loaded model: CBraMod (4.9M param pretrained transformer) or EEGNet (25K param custom CNN)
- Shows architecture details, input specs (raw EEG vs DE features), dual-head outputs (arousal + valence), and training metrics
- Demonstrates transparency — the user can see exactly what model is making predictions

**Follow-up:**

> "How does it classify emotions from brain waves?"

- The agent explains the classification pipeline: EEG signal → preprocessing → model inference → arousal/valence binary classification → emotion quadrant mapping
- References the specific model architecture (criss-cross transformer for CBraMod, spatial/temporal convolutions for EEGNet)

### Session Deep Dive + New Music Discovery

The full end-to-end retrieval demo: analyze a session and discover new matching music in a single chat thread, with both inline visualizations rendered side by side.

**Step 1 — Analyze the session:**

> "Analyze Session 07"

- Agent calls `analyze_session` — `<SessionVisualization>` renders inline with the Trajectory tab (default, animated path through Russell's affect space) and Timeline tab, plus the stacked frequency-band-power chart below
- The agent narrates the emotional arc using `trajectory_summary` fields (dwell fractions per quadrant, transition count, centroid, path length)

**Step 2 — Retrieve matching tracks in the same turn:**

> "Now find me new songs that sound like how I was feeling"

- Agent picks `retrieve_tracks_from_brain_state` (not a quadrant-filter tool) — `<RetrievedTracksPanel>` renders beneath the new tool call with ranked tracks and similarity bars
- Both panels from Steps 1 and 2 stay visible in the transcript — the trajectory chart shows what you *experienced*, the retrieval list shows what else the model thinks would match that experience

**Step 3 — Act on the retrieval (optional):**

> "Build a playlist from the top 5 of those"

- Agent proposes a playlist name and the top-5 track list, waits for your confirmation, then calls `build_mood_playlist` with `user_confirmed=True`
- If Spotify is connected, the playlist is created live; otherwise saved locally

**What to watch for across the session:** The two inline visualizations side by side are the feature's best screenshot. Notice that retrieved tracks are typically ones the user has never heard in this database — that's the fundamental difference from `find_relaxing_tracks`, which filters the EEG-labeled history. If similarity scores are weak across the board (near zero), the contrastive checkpoint is likely underfit — run a full Modal training pass (`modal run backend/scripts/modal_train.py --command train-contrastive`) for a real checkpoint.

---

## Quick Demos

**List sessions:** *"Show me all EEG sessions"* — renders the `<SessionListPanel>` card grid with stable Session NN labels, derived dominant-state captions, and quadrant distribution bars.

**Analyze a session:** *"Analyze Session 04"* — full breakdown with per-segment timeline, band powers, and track associations; the agent resolves the label to a UUID internally.

**Explain brain state:** *"What was my brain doing during Session 09?"* — plain-language neuroscience narrative with frequency band context.

**Compare sessions:** *"Compare Session 03 and Session 18"* — side-by-side with arousal/valence deltas and dominant state changes.

**Find mood tracks:** *"Find tracks that triggered excited brain states"* — EEG-derived recommendations by emotion quadrant.

**Retrieve matching tracks (contrastive):** *"Find songs that match my brain state in Session 07"* — contrastive EEG↔CLAP retrieval from the pgvector audio index; renders the similarity-bar track list inline with preview playback. Distinct from the quadrant-filter tools above: these may be tracks the user has never heard.

**Build playlist:** *"Build me a stressed-out playlist"* — mood playlist from brain data with user confirmation gate.

**Search Spotify:** *"Search Spotify for Radiohead"* — public search, no connection required.

**Track details:** *"Get info on that track"* — Spotify metadata lookup after a search.

**Set brain context:** *"Set my brain context to relaxed with arousal 0.2 and valence 0.8"* — manual context override; badge appears in header.

**Model info:** *"What model are you using?"* — architecture, parameters, and training metrics.

---

## Future Features

These workflows will become available as upcoming roadmap features are implemented. They're included here to illustrate the vision.

### Live BCI Classification *(Phase 3)*

> "Start monitoring my brain while I listen to Spotify"

- Muse 2 headband streams 4-channel EEG over Bluetooth via muselsl/pylsl
- Real-time classification runs every 4-second segment during Spotify playback
- WebSocket pushes live brain state updates to the frontend
- Brain context badge updates in real-time as mood shifts

**Follow-up:**

> "How has my brain responded to the last 3 tracks?"

- Agent queries the live session's accumulated segments
- Shows per-track brain state breakdown as it happens

### Adaptive Playlist *(Phase 3)*

> "Play music and adjust based on my brain state — I want to stay relaxed"

- Live classification detects brain state drift (relaxed → stressed)
- Agent automatically queues calmer tracks when arousal spikes
- Session is recorded for later analysis

### Personalized Models *(Phase 4)*

> "Fine-tune the model on my brain data"

- Few-shot fine-tuning: freeze the pretrained encoder, train the classification head on ~10 minutes of the user's labeled EEG data
- Future sessions use the personalized model for better individual accuracy
- Calibration flow: 5-minute baseline recording to establish the user's neural patterns

### Attention Visualization *(Phase 4)*

> "Show me which brain regions drove the prediction for segment 12"

- Extract transformer attention weights from CBraMod
- Render channel/timepoint importance maps showing which EEG channels and moments most influenced the arousal/valence classification
- Topographic brain maps visualize electrode activation patterns on a 2D scalp layout

### Cross-Session Trend Analysis *(Phase 4)*

> "How have my brain responses to music changed over the last month?"

- Extract pretrained embeddings across sessions
- Track trajectories in embedding space over time
- Identify long-term shifts in emotional response patterns

---

## Presenter Tips

- **Start fresh:** Each workflow assumes a new chat thread. Click the new chat button in the sidebar.
- **Reference by label, not UUID:** After `list_sessions` runs, refer to sessions by their **Session NN** label (e.g. "analyze Session 07"). The agent resolves the label to the underlying UUID internally — UUIDs, participant IDs, and recorded-at timestamps are intentionally hidden from the chat surface.
- **Inline visualizations are the screenshot:** `list_sessions` renders the `<SessionListPanel>` card grid; `analyze_session` renders `<SessionVisualization>` (animated trajectory + timeline + band-power chart); `retrieve_tracks_from_brain_state` renders `<RetrievedTracksPanel>` with similarity bars and inline previews. The raw tool-output JSON is hidden behind a collapsible — expand it only if you specifically want to show the reasoning pipeline.
- **Expand tool calls:** Click the collapsible tool call indicators to show input parameters and raw output JSON. This demonstrates the agent's reasoning pipeline.
- **Brain context persistence:** Refresh the page mid-workflow to show that brain context survives page reloads — it's stored in the thread's database column, not browser state.
- **Spotify connection:** Workflows using `get_listening_history`, `get_my_playlists`, `get_my_saved_tracks`, or `add_tracks_to_playlist` require Spotify to be connected in Settings. The tools are hidden from the agent when not connected.
- **Contrastive retrieval prereqs:** `retrieve_tracks_from_brain_state` requires a populated `track_audio_embeddings` index (run `seed-track-index`) and a contrastive checkpoint (`train-contrastive` locally or via Modal). If either is missing, the tool returns a structured payload with operator-fix instructions that the agent relays verbatim — no silent hallucinations.
- **User confirmation gates:** `build_mood_playlist` and `add_tracks_to_playlist` always ask for confirmation before executing. The agent will never create or modify playlists without explicit approval.
- **Dark mode:** Toggle the theme to show the full dark mode experience — tool calls, badges, and chat all adapt.
