# CortexDJ Roadmap

## Phase 2: Real EEG Datasets

- DEAP dataset integration (32 participants, music + emotion labels, requires registration)
- SEED dataset support (15 participants, film clips, freely available)
- AMIGOS dataset support (40 participants, audio stimuli)
- Dataset-agnostic data loader with format autodetection (.dat, .mat, .edf)
- MNE-Python raw data preprocessing (ICA artifact removal, re-referencing)

### Pretrained Model Opportunity

- Evaluate CBraMod and REVE on DEAP/SEED/AMIGOS — pretrained on massive EEG corpora (TUEG, 92 datasets respectively), likely outperform training from scratch on 32-participant datasets
- Implement `from_pretrained()` + `reset_head(2)` fine-tuning loop alongside existing training pipeline
- Benchmark: pretrained fine-tuned vs. current EEGNetClassifier on DEAP arousal/valence accuracy
- See [Pretrained Models Analysis](pretrained-models-analysis.md) for model selection rationale

## Phase 3: Live BCI Device Integration

- Muse 2 headband support via muselsl/pylsl
- Real-time EEG stream ingestion endpoint
- Live classification during Spotify playback
- WebSocket stream for live brain state updates to frontend
- EEG waveform visualization component (time-series + topographic map)

### Pretrained Model Opportunity

- Muse 2 has only 4 EEG channels — current 32-channel EEGNetClassifier cannot work without complete retraining
- CBraMod, REVE, and LUNA support arbitrary channel counts, enabling direct transfer from 32-channel training to 4-channel inference
- Evaluate real-time inference latency of CBraMod (4.9M params) for live classification during Spotify playback
- See [Pretrained Models Analysis](pretrained-models-analysis.md) for channel-flexible model comparison

## Phase 4: Advanced ML

- CNN-Transformer hybrid model (EEGNet backbone + Transformer encoder)
- Personalized models — fine-tune per user from their session history
- Cross-session trend analysis (how brain responses change over time)
- Transfer learning from DEAP pre-training to individual users
- Attention visualization — which channels/timepoints drive predictions

### Pretrained Model Opportunity

- **CNN-Transformer hybrid** — CBraMod, EEGPT, BENDR are production-ready transformer architectures; no need to build from scratch
- **Transfer learning** — `from_pretrained()` + `reset_head()` + fine-tune is the standard braindecode pattern
- **Personalized models** — few-shot fine-tuning of pretrained encoder on individual user data (freeze encoder, train head on ~10 min of data)
- **Attention visualization** — extract transformer attention weights from pretrained models for channel/timepoint importance maps
- **Cross-session trends** — use `return_features=True` to extract embeddings; track trajectories across sessions in embedding space
- See [Pretrained Models Analysis](pretrained-models-analysis.md) for API patterns and migration architecture

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
