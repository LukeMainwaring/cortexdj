# CortexDJ Roadmap

> Forward-looking plan. Shipped features are documented in [README.md](../README.md).
> Pretrained model details: [pretrained-models-analysis.md](pretrained-models-analysis.md)

## Demo

- Public demo mode — pre-loaded DEAP sessions anyone can explore without EEG hardware or Spotify auth; reuses seeded sessions and the shipped `SessionVisualization`

## Visualization

- Emotion trajectory — 2D arousal/valence scatter animated over time, tracing a path through the four emotion quadrants (extends `SessionVisualization`, which currently renders them as separate time-series lines)
- True time-frequency spectrogram — per-channel STFT heatmap (frequency on y-axis, color = power), distinct from the stacked band-power area chart already shipped
- Session comparison dashboard — side-by-side `SessionVisualization` renders for two session IDs
- Export reports — PDF/image export of session analysis for sharing

## Phase 2: Real EEG Datasets

- SEED dataset support (15 participants, film clips, freely available)
- AMIGOS dataset support (40 participants, audio stimuli)
- Dataset-agnostic data loader with format autodetection (.dat, .mat, .edf)
- MNE-Python raw data preprocessing (ICA artifact removal, re-referencing)
- Evaluate REVE on DEAP — pretrained on 92 datasets, 25K subjects; may outperform CBraMod

## Phase 3: Live BCI Device Integration

- Muse 2 headband support via muselsl/pylsl
- Synthetic 4-channel EEG generator for hardware-free Muse pipeline development (4 channels at 256Hz, TP9/AF7/AF8/TP10 montage)
- Real-time EEG stream ingestion endpoint
- Live classification during Spotify playback
- WebSocket stream for live brain state updates to frontend (the existing inline `<SessionVisualization>` chart is the target render surface — point the live stream at it)
- Topographic brain-map view (per-channel scalp heatmap) — complements the existing arousal/valence + band-power timeline
- Evaluate CBraMod real-time inference latency for live classification (<500ms target)
- Benchmark 32ch->4ch transfer accuracy degradation (CBraMod, REVE, LUNA)
- Calibration flow — 5-minute baseline recording to personalize the classifier per user (prereq for adaptive playlist quality)
- Session recording during Spotify playback — persist live EEG + track metadata for later analysis (ground truth for the Phase 5 "Now Playing" correlation item)
- Adaptive playlist — skip/queue tracks based on live brain state (e.g., stressed -> switch to a calmer track)

## Phase 4: Advanced ML

- EEG data augmentation — Gaussian noise injection, temporal jittering/random cropping, channel dropout
- CNN-Transformer hybrid (CBraMod, EEGPT, BENDR are production-ready options)
- Personalized models — few-shot fine-tuning of pretrained encoder on individual user data
- Cross-session trend analysis — track embedding trajectories across sessions
- Transfer learning from DEAP pre-training to individual users
- Attention visualization — extract transformer attention weights for channel/timepoint importance maps
- Model ensemble — combine EEGNet (DE features) and CBraMod (raw EEG) predictions
- Multi-task learning — add discrete emotion heads (happy/sad/angry/fearful) alongside existing arousal/valence heads (DEAP ships discrete labels; incremental dual-head extension)

## Phase 5: Spotify Deep Integration

- Real-time "Now Playing" correlation — classify brain state while user listens
- Recommendation engine combining brain-state preferences with Spotify audio features
- Library analysis — scan user's saved tracks and predict brain-state compatibility
- Collaborative filtering — "users with similar brain patterns also liked..."
- Audio feature correlation — correlate Spotify energy/danceability/acousticness with arousal/valence from `EegSegment` (SQL join on existing tables; feeds the recommendation engine)
- Genre brain mapping — aggregate brain states by Spotify genre

## Phase 6: Platform

- User authentication (OAuth or magic link)
- Multi-user session management
- Cloud deployment (Railway/Render)
- CD pipeline with model versioning
- Mobile companion app for BCI device pairing

## Technical Debt

- Expand pytest suite — add async integration tests with model mocking
- Add WebSocket support for real-time brain state streaming
- Frontend test infrastructure (currently zero coverage on `frontend/components/`)
