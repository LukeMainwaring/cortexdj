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
- **Contrastive learning** — learn embeddings where similar brain states cluster together (SupCon on EEG) — or leverage BENDR/SignalJEPA which already encode contrastive structure
- **Cross-dataset transfer** — pre-train on DEAP, fine-tune on SEED, evaluate generalization — REVE's 4D positional encoding specifically designed for cross-configuration transfer

## Pretrained Foundation Models

Ideas enabled by braindecode 1.4.0's pretrained EEG encoders. See [Pretrained Models Analysis](pretrained-models-analysis.md) for full technical details.

- **Zero-shot emotion classification** — use REVE's linear probing on emotion datasets without any CortexDJ-specific training data
- **Embedding-based mood tracking** — extract pretrained embeddings (`return_features=True`) across a listening session; cluster and visualize mood trajectory in embedding space
- **Cross-device transfer** — train on DEAP (32ch lab EEG), deploy on Muse 2 (4ch consumer) using CBraMod/LUNA's channel-invariant architecture
- **Foundation model A/B testing** — run multiple pretrained models in parallel, compare which best predicts user's Spotify skip behavior as ground truth
- **Model distillation** — distill large pretrained model (REVE/BENDR) into a tiny model for real-time mobile inference
- **HuggingFace model sharing** — `push_to_hub()` to share CortexDJ fine-tuned emotion models; `from_pretrained()` to load community models
- **Ensemble stacking** — combine embeddings from multiple pretrained models (e.g., CBraMod + SignalJEPA) as input to a lightweight emotion classifier
- **Curriculum fine-tuning** — progressive unfreezing: first train only the head, then unfreeze top transformer layers, then full model
- **Temporal context from embeddings** — feed sequence of per-segment pretrained embeddings into a lightweight temporal model (LSTM/Transformer) for session-level emotion trajectory prediction

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
- **Model marketplace** — share pre-trained models for different use cases — braindecode's `push_to_hub()` / `from_pretrained()` provides the HuggingFace integration backbone
- **Research mode** — export data in BIDS format for academic use
