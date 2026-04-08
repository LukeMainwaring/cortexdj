# CortexDJ Feature Brainstorm

> Idea backlog. Promoted ideas move to [ROADMAP.md](ROADMAP.md).

## Real-Time BCI

- **Adaptive playlist** — skip or queue tracks based on live brain state (stressed -> switch to calmer music)
- **Session recording** — record live EEG during Spotify listening, save to database for later analysis
- **Calibration flow** — 5-minute baseline recording to personalize the model

## Advanced ML

- **Multi-task learning** — predict arousal, valence, AND specific emotions (happy, sad, angry, fearful) simultaneously
- **Temporal context** — LSTM or Transformer layer to capture how brain states evolve within a session
- **Contrastive learning** — learn embeddings where similar brain states cluster (SupCon on EEG) — or leverage BENDR/SignalJEPA which already encode contrastive structure
- **Foundation model A/B testing** — run multiple pretrained models in parallel, compare which best predicts user's Spotify skip behavior as ground truth
- **Ensemble stacking** — combine embeddings from multiple pretrained models (e.g., CBraMod + SignalJEPA) as input to a lightweight emotion classifier

## Spotify Intelligence

- **Audio feature correlation** — correlate Spotify audio features (energy, danceability, acousticness) with brain states
- **Discovery mode** — play new tracks and classify brain response to build a preference model
- **Genre brain mapping** — build a map of which genres activate which brain states
- **Social playlists** — "music that makes both of us relaxed" from two users' brain data
- **Time-of-day patterns** — learn that user prefers calming music in the morning, energizing at noon

## Visualization

- **Time-frequency spectrograms** — show how band powers evolve over a session
- **Emotion trajectory** — arousal/valence scatter plot animated over time
- **Session comparison dashboard** — side-by-side session visualizations
- **Export reports** — PDF/image export of session analysis for sharing

## Platform

- **Public demo mode** — pre-loaded sessions anyone can explore without EEG hardware
- **API access** — REST API for third-party integrations (other music apps, research tools)
- **Model marketplace** — share pre-trained models via braindecode's `push_to_hub()` / `from_pretrained()` HuggingFace integration
- **Research mode** — export data in BIDS format for academic use
