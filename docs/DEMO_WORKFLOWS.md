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

> "What was my brain doing during session [ID]?"

The agent translates raw EEG classification data into an accessible narrative — explaining dominant emotional states, notable transitions, and the underlying neural signatures. It references specific frequency bands (alpha, beta, theta) and explains what they indicate about cognitive and emotional processing.

**What to watch for:**

- "Interpreting brain activity" spinner → checkmark
- Response includes frequency band explanations: "Strong alpha activity (8-14 Hz) indicates relaxed alertness"
- Emotion quadrant mapping: arousal/valence scores mapped to relaxed, calm, excited, or stressed
- State distribution breakdown: "65% relaxed, 20% calm, 10% excited, 5% stressed"
- Educational without being clinical — the agent makes neuroscience approachable

### Cross-Session Brain Pattern Comparison

> "Compare session [ID1] and session [ID2] — what changed?"

Side-by-side comparison of brain responses across two different listening sessions. Reveals how the same person's brain states differ across different music or different days.

**What to watch for:**

- "Comparing sessions" spinner → checkmark
- Delta values: arousal and valence differences between sessions
- Dominant state comparison: did the participant shift from stressed to relaxed?
- The agent interprets the deltas — "Your brain was significantly more relaxed during session 2, with a 0.3 drop in arousal"

### Persistent Brain Context

> "Set my brain context to session [ID]"

Then in follow-up messages:

> "Build me a playlist"

The agent persists brain state context (dominant mood, arousal, valence) in the conversation thread. All subsequent interactions are informed by this context — playlist recommendations, analysis framing, and the agent's tone all adapt.

**What to watch for:**

- Brain context badge appears in the chat header with pills for mood, arousal, and valence
- Refresh the page — the badge reappears because context is persisted in the thread's database column, not browser state
- Follow-up requests automatically reference the active brain context without the user repeating it

---

## End-to-End Workflows

### Full Session Deep Dive

A 4-step workflow demonstrating the complete EEG analysis pipeline.

**Step 1 — Discover sessions:**

> "Show me my EEG sessions"

- Agent calls `list_sessions` — "Loading EEG sessions" spinner → checkmark
- Results list sessions with participant IDs, timestamps, and segment counts
- Sessions come from the DEAP dataset (32 participants, 40 music video trials each)

**Step 2 — Analyze a session:**

> "Analyze session [paste an ID from the list]"

- Agent calls `analyze_session` — "Analyzing brain states" spinner → checkmark
- Response includes:
  - Session metadata (participant, duration, recording source)
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

---

## Quick Demos

**List sessions:** *"Show me all EEG sessions"* — loads session index with participant info and timestamps.

**Analyze a session:** *"Analyze session [ID]"* — full breakdown with per-segment timeline, band powers, and track associations.

**Explain brain state:** *"What was my brain doing during session [ID]?"* — plain-language neuroscience narrative with frequency band context.

**Compare sessions:** *"Compare sessions [ID1] and [ID2]"* — side-by-side with arousal/valence deltas and dominant state changes.

**Find mood tracks:** *"Find tracks that triggered excited brain states"* — EEG-derived recommendations by emotion quadrant.

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
- **Reference by ID:** Session IDs from `list_sessions` are used in follow-up prompts. Copy-paste from the results.
- **Expand tool calls:** Click the collapsible tool call indicators to show input parameters and raw output JSON. This demonstrates the agent's reasoning pipeline.
- **Brain context persistence:** Refresh the page mid-workflow to show that brain context survives page reloads — it's stored in the thread's database column, not browser state.
- **Spotify connection:** Workflows using `get_listening_history`, `get_my_playlists`, `get_my_saved_tracks`, or `add_tracks_to_playlist` require Spotify to be connected in Settings. The tools are hidden from the agent when not connected.
- **User confirmation gates:** `build_mood_playlist` and `add_tracks_to_playlist` always ask for confirmation before executing. The agent will never create or modify playlists without explicit approval.
- **Dark mode:** Toggle the theme to show the full dark mode experience — tool calls, badges, and chat all adapt.
