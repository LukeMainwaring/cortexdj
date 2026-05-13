# CortexDJ Roadmap

> Forward-looking plan. Shipped features are documented in [README.md](../README.md).
> Pretrained model details: [pretrained-models-analysis.md](pretrained-models-analysis.md). Dataset catalog: [datasets-analysis.md](datasets-analysis.md).

## Phase 1: Near-term polish

- Public demo mode — pre-loaded DEAP sessions anyone can explore without EEG hardware or Spotify auth; reuses seeded sessions and the shipped `SessionVisualization`
- True time-frequency spectrogram — per-channel STFT heatmap (frequency on y-axis, color = power), distinct from the stacked band-power area chart already shipped
- Session comparison dashboard — side-by-side `SessionVisualization` renders for two session IDs

## Phase 2: Real EEG Datasets

- SEED dataset support (15 participants, film clips) — adds discrete emotion labels (positive/neutral/negative) alongside DEAP's continuous valence/arousal
- AMIGOS dataset support (40 participants, short + long video clips)
- DREAMER + MAHNOB-HCI + FACED via [EEGain](https://github.com/EmotionLab/EEGain) — leave-one-dataset-out generalization eval; FACED specifically enables direct REVE-comparison (see [datasets-analysis.md](datasets-analysis.md) Tier A / Path A)
- Adopt EEGain as the unified dataset layer (DEAP/SEED/SEED-IV/DREAMER/MAHNOB-HCI/AMIGOS preprocessing standardized to 128 Hz / 4-s windows); plug CBraMod heads on top — supersedes the earlier "dataset-agnostic loader" line item
- MNE-Python raw data preprocessing — ICA artifact removal, re-referencing

## Phase 3: Live BCI Device Integration

- Muse 2 headband support via muselsl/pylsl
- Synthetic 4-channel EEG generator for hardware-free Muse pipeline development (TP9/AF7/AF8/TP10 @ 256 Hz)
- Real-time EEG stream ingestion endpoint
- Live classification during Spotify playback
- WebSocket stream for live brain state updates — targets the existing inline `<SessionVisualization>` chart
- CBraMod real-time inference latency benchmark (<500ms target for 4s segments)
- 32ch→4ch transfer accuracy benchmark (CBraMod, REVE, LUNA) — Muse 2 montage masked on DEAP
- Per-user calibration — 5-min baseline recording; prereq for adaptive playlist quality
- Adaptive playlist — skip/queue tracks based on live brain state

## Phase 4: Advanced ML

- EEG data augmentation — Gaussian noise, temporal jittering, channel dropout
- Extend the autoresearch harness (`backend/autoresearch/`, currently EEGNet-only) to the contrastive encoder and multi-GPU fan-out; warm the Modal image to cut the ~15-min cold-start on each run
- Evaluate pretrained encoders beyond CBraMod — EEGPT, BENDR, REVE, LUNA — as drop-in backbone replacements (note: REVE's emotion benchmark is FACED, not DEAP, so DEAP numbers need direct measurement)
- Personalized fine-tuning — few-shot adaptation of the CBraMod encoder to individual users on top of DEAP fine-tuning
- Cross-session trend analysis — track embedding trajectories across sessions
- Attention visualization — transformer attention weights as channel/timepoint importance maps
- Model ensemble — combine EEGNet (DE features) and CBraMod (raw EEG) predictions
- Discrete emotion classification — add a positive/neutral/negative head once SEED support lands (Phase 2); DEAP itself ships only continuous valence/arousal/dominance/liking
- (Stretch) HBN-EEG intermediate-pretraining stage — masked-prediction or relative-positioning on 3,000-subject movie-watching data, between public CBraMod backbone and DEAP fine-tuning. Gated on Phase 2 / EEG↔CLAP work identifying a representation-quality bottleneck. See [datasets-analysis.md](datasets-analysis.md) Tier C / Path C.

## Phase 5: Spotify Deep Integration

- "Now Playing" correlation — persist live EEG + track metadata during Spotify playback, then classify brain state per track (depends on Phase 3 live-stream endpoint)
- Library analysis — scan user's saved tracks and predict brain-state compatibility
- Collaborative filtering — "users with similar brain patterns also liked..."
- Audio feature correlation — correlate Spotify energy/danceability/acousticness with arousal/valence from `EegSegment`
- Genre brain mapping — aggregate brain states by Spotify genre

### Deferred research: EEG↔CLAP contrastive retrieval

An EEG-to-audio contrastive encoder was wired end-to-end (`ml/contrastive*.py`, `services/retrieval.py`, `track_audio_embeddings` pgvector index) but does not produce usable retrieval signal at DEAP scale — within-subject cross-trial eval also plateaued at the per-pool uniform-random baseline, ruling out subject transfer as the bottleneck. Four-second EEG windows encode mood/arousal/attention, not the timbral and harmonic track identity that dominates LAION-CLAP audio embeddings.

Directions that would make this tractable:

- Per-user calibration + continual learning from explicit (thumbs) and implicit (skip/replay) in-app feedback
- Longer-integration EEG windows (30s+) or multimodal fusion (skin conductance, fNIRS) to raise per-inference SNR
- Retrieve against emotion-quadrant centroids rather than specific track identities, aligning the target with what EEG actually encodes
- Re-train on music-listening datasets (MUSIN-G, NMED-T, NMED-H, OpenMIIR) where EEG was recorded during music perception and full-song audio is provided alongside — replaces DEAP's video-clip-with-30s-iTunes-preview chain at every weak link. See [datasets-analysis.md](datasets-analysis.md) Tier B / Path B.

## Phase 6: Platform

- User authentication (OAuth or magic link)
- Multi-user session management
- Cloud deployment (Railway/Render) with model versioning

## Technical Debt

- Expand pytest suite — async integration tests with model mocking
- Frontend test infrastructure — zero current coverage on `frontend/components/`
