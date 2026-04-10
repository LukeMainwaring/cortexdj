# CortexDJ Roadmap

> Forward-looking plan. Shipped features are documented in [README.md](../README.md).
> Pretrained model details: [pretrained-models-analysis.md](pretrained-models-analysis.md)

## Demo

- Public demo mode — pre-loaded DEAP sessions anyone can explore without EEG hardware or Spotify auth; reuses seeded sessions and the shipped `SessionVisualization`

## Visualization

- True time-frequency spectrogram — per-channel STFT heatmap (frequency on y-axis, color = power), distinct from the stacked band-power area chart already shipped
- Session comparison dashboard — side-by-side `SessionVisualization` renders for two session IDs
- Topographic scalp heatmap — per-channel band-power interpolated over the 10-20 montage (DEAP's 32-channel recordings only; not meaningful for Muse 2's 4 electrodes)
- Export reports — PDF/image export of session analysis for sharing

## Phase 2: Real EEG Datasets

- SEED dataset support (15 participants, film clips; available on request from BCMI/SJTU) — adds discrete emotion labels (positive/neutral/negative) alongside DEAP's continuous valence/arousal
- AMIGOS dataset support (40 participants, short + long video clips)
- Dataset-agnostic data loader with format autodetection (.dat, .mat, .edf)
- MNE-Python raw data preprocessing (ICA artifact removal, re-referencing)
- Evaluate REVE on DEAP — pretrained on 60K+ hours / 92 datasets / 25K subjects; note REVE's emotion benchmark is FACED, not DEAP, so DEAP numbers need direct measurement

## Phase 3: Live BCI Device Integration

- Muse 2 headband support via muselsl/pylsl
- Synthetic 4-channel EEG generator for hardware-free Muse pipeline development (4 channels at 256Hz, TP9/AF7/AF8/TP10 montage)
- Real-time EEG stream ingestion endpoint
- Live classification during Spotify playback
- WebSocket stream for live brain state updates to frontend (the existing inline `<SessionVisualization>` chart is the target render surface — point the live stream at it)
- Evaluate CBraMod real-time inference latency for live classification (<500ms target)
- Benchmark 32ch->4ch transfer accuracy degradation (CBraMod, REVE, LUNA)
- Calibration flow — 5-minute baseline recording to personalize the classifier per user (prereq for adaptive playlist quality)
- Session recording during Spotify playback — persist live EEG + track metadata for later analysis (ground truth for the Phase 5 "Now Playing" correlation item)
- Adaptive playlist — skip/queue tracks based on live brain state (e.g., stressed -> switch to a calmer track)

## Phase 4: Advanced ML

- EEG data augmentation — Gaussian noise injection, temporal jittering/random cropping, channel dropout
- Evaluate additional pretrained encoders beyond CBraMod — EEGPT, BENDR, REVE — as drop-in replacements for the current backend
- Personalized fine-tuning — few-shot adaptation of the CBraMod encoder (pretrained on TUEG) to individual users on top of DEAP fine-tuning
- Cross-session trend analysis — track embedding trajectories across sessions
- Attention visualization — extract transformer attention weights for channel/timepoint importance maps
- Model ensemble — combine EEGNet (DE features) and CBraMod (raw EEG) predictions
- Discrete emotion classification once SEED support lands (Phase 2) — add a positive/neutral/negative head alongside the existing arousal/valence heads (DEAP itself only ships continuous valence/arousal/dominance/liking ratings, so this depends on a dataset with categorical labels)

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
