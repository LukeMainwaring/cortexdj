# CortexDJ Feature Brainstorm

Ideas for future development, organized by theme.

## Real-Time BCI Integration

- **Muse 2 headband** via muselsl/pylsl — stream 4-channel EEG over Bluetooth
- **Live classification overlay** — show brain state in real-time while Spotify plays
- **Adaptive playlist** — skip or queue tracks based on live brain state (if user is getting stressed, switch to calmer music)
- **Session recording** — record live EEG during Spotify listening, save to database for later analysis
- **Calibration flow** — 5-minute baseline recording to personalize the model

## Advanced ML

- **Attention maps** — visualize which EEG channels and timepoints drive the model's predictions
- **Subject-adaptive training** — fine-tune the base model per user with ~10 minutes of labeled data
- **Multi-task learning** — predict arousal, valence, AND specific emotions (happy, sad, angry, fearful) simultaneously
- **Temporal context** — LSTM or Transformer layer to capture how brain states evolve within a session
- **Contrastive learning** — learn embeddings where similar brain states cluster together (SupCon on EEG)
- **Cross-dataset transfer** — pre-train on DEAP, fine-tune on SEED, evaluate generalization

## Spotify Intelligence

- **Audio feature correlation** — correlate Spotify audio features (energy, danceability, acousticness) with brain states
- **Discovery mode** — play new tracks and classify brain response to build a preference model
- **Genre brain mapping** — build a map of which genres activate which brain regions/states
- **Social playlists** — "music that makes both of us relaxed" from two users' brain data
- **Time-of-day patterns** — learn that user prefers calming music in the morning, energizing at noon

## Visualization

- **Topographic brain maps** — 2D scalp map showing electrode activation patterns
- **Time-frequency spectrograms** — show how band powers evolve over a session
- **Emotion trajectory** — arousal/valence scatter plot animated over time
- **Session comparison dashboard** — side-by-side session visualizations
- **Export reports** — PDF/image export of session analysis for sharing

## Platform

- **User accounts** — OAuth login, personal model storage, session history
- **Public demo mode** — pre-loaded sessions anyone can explore without EEG hardware
- **API access** — REST API for third-party integrations (other music apps, research tools)
- **Model marketplace** — share pre-trained models for different use cases
- **Research mode** — export data in BIDS format for academic use
