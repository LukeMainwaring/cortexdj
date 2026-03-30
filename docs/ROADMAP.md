# CortexDJ Roadmap

## Phase 2: Real EEG Datasets

- DEAP dataset integration (32 participants, music + emotion labels, requires registration)
- SEED dataset support (15 participants, film clips, freely available)
- AMIGOS dataset support (40 participants, audio stimuli)
- Dataset-agnostic data loader with format autodetection (.dat, .mat, .edf)
- MNE-Python raw data preprocessing (ICA artifact removal, re-referencing)

## Phase 3: Live BCI Device Integration

- Muse 2 headband support via muselsl/pylsl
- Real-time EEG stream ingestion endpoint
- Live classification during Spotify playback
- WebSocket stream for live brain state updates to frontend
- EEG waveform visualization component (time-series + topographic map)

## Phase 4: Advanced ML

- CNN-Transformer hybrid model (EEGNet backbone + Transformer encoder)
- Personalized models — fine-tune per user from their session history
- Cross-session trend analysis (how brain responses change over time)
- Transfer learning from DEAP pre-training to individual users
- Attention visualization — which channels/timepoints drive predictions

## Phase 5: Spotify Deep Integration
- Real-time "Now Playing" correlation — classify brain state while user listens
- Recommendation engine combining brain-state preferences with Spotify audio features
- Library analysis — scan user's saved tracks and predict brain-state compatibility
- Collaborative filtering — "users with similar brain patterns also liked..."

## Phase 6: Platform

- User authentication (OAuth or magic link)
- Multi-user session management
- Cloud deployment (Railway/Render)
- CI/CD pipeline with model versioning
- Mobile companion app for BCI device pairing

## Dataset Action Items


| Dataset | Size    | Participants | Stimulus       | Labels                                    | Access                |
| ------- | ------- | ------------ | -------------- | ----------------------------------------- | --------------------- |
| DEAP    | ~6.5 GB | 32           | Music videos   | Arousal, valence, liking, dominance (0-9) | Registration required |
| SEED    | ~1-2 GB | 15           | Film clips     | Positive, negative, neutral               | Freely available      |
| AMIGOS  | ~2-3 GB | 40           | Video extracts | Arousal, valence                          | Registration required |


## Technical Debt

- Add comprehensive pytest suite with model mocking
- Add WebSocket support for real-time brain state streaming
- Frontend EEG visualization (recharts or d3 for time-series, brain topomaps)
