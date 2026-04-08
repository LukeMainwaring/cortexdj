# CortexDJ Roadmap

## Phase 2: Real EEG Datasets

- ~~DEAP dataset integration (32 participants, music + emotion labels)~~ (shipped)
- ~~DEAP data loader with baseline stripping, feature/raw modes, participant tracking~~ (shipped)
- ~~Leave-one-subject-out (LOSO) cross-validation and participant-grouped CV~~ (shipped)
- SEED dataset support (15 participants, film clips, freely available)
- AMIGOS dataset support (40 participants, audio stimuli)
- Dataset-agnostic data loader with format autodetection (.dat, .mat, .edf)
- MNE-Python raw data preprocessing (ICA artifact removal, re-referencing)

### Pretrained Model Opportunity

- ~~CBraMod integration via braindecode — `PretrainedDualHead` wrapper with freeze/unfreeze, two-phase fine-tuning~~ (shipped)
- ~~Dual model backends: `EEG_MODEL_BACKEND=eegnet` (DE features) or `EEG_MODEL_BACKEND=cbramod` (raw EEG)~~ (shipped)
- ~~`compare-models` CLI for benchmarking EEGNet vs CBraMod on DEAP~~ (shipped)
- Evaluate REVE on DEAP — pretrained on 92 datasets, 25K subjects; may outperform CBraMod
- See [Pretrained Models Analysis](pretrained-models-analysis.md) for model selection rationale

## Phase 3: Live BCI Device Integration

- Muse 2 headband support via muselsl/pylsl
- Synthetic 4-channel EEG generator for hardware-free Muse pipeline development (4 channels at 256Hz, TP9/AF7/AF8/TP10 montage)
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
- ~~Library access — agent tools for browsing playlists, saved tracks, and searching Spotify~~ (shipped)
- ~~Playlist management — add tracks to existing playlists with user confirmation gates~~ (shipped)
- ~~Structured error handling — HTTP status-aware Spotify error responses~~ (shipped)
- Real-time "Now Playing" correlation — classify brain state while user listens
- Recommendation engine combining brain-state preferences with Spotify audio features
- Library analysis — scan user's saved tracks and predict brain-state compatibility
- Collaborative filtering — "users with similar brain patterns also liked..."

## Phase 6: Platform

- User authentication (OAuth or magic link)
- Multi-user session management
- Cloud deployment (Railway/Render)
- ~~CI pipeline (GitHub Actions: pre-commit + pytest)~~ (shipped)
- CD pipeline with model versioning
- Mobile companion app for BCI device pairing

## Dataset Action Items


| Dataset | Size    | Participants | Stimulus       | Labels                                    | Access                |
| ------- | ------- | ------------ | -------------- | ----------------------------------------- | --------------------- |
| DEAP    | ~6.5 GB | 32           | Music videos   | Arousal, valence, liking, dominance (0-9) | Freely available (Kaggle mirror) |
| SEED    | ~1-2 GB | 15           | Film clips     | Positive, negative, neutral               | Freely available      |
| AMIGOS  | ~2-3 GB | 40           | Video extracts | Arousal, valence                          | Registration required |


## Technical Debt

- Expand pytest suite — current tests cover preprocessing and dataset utilities; add async integration tests with model mocking
- Add WebSocket support for real-time brain state streaming
- Frontend EEG visualization (recharts or d3 for time-series, brain topomaps)
