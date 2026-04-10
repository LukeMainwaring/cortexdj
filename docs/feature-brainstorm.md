# CortexDJ Feature Brainstorm

> Idea backlog. Promoted ideas move to [ROADMAP.md](ROADMAP.md).

## Advanced ML

- **Temporal context** — LSTM or Transformer layer to capture how brain states evolve within a session (partially covered by the Phase 4 CNN-Transformer hybrid item in ROADMAP.md)
- **Contrastive learning** — learn embeddings where similar brain states cluster (SupCon on EEG) — or leverage BENDR/SignalJEPA which already encode contrastive structure
- **Foundation model A/B testing** — run multiple pretrained models in parallel, compare which best predicts user's Spotify skip behavior as ground truth (research experiment — belongs in `docs/pretrained-models-analysis.md` rather than the roadmap)
- **Ensemble stacking** — combine embeddings from multiple pretrained models (e.g., CBraMod + SignalJEPA) as input to a lightweight emotion classifier (covered by the Phase 4 "Model ensemble" item)

## Spotify Intelligence

- **Discovery mode** — play new tracks and classify brain response to build a preference model (blocks on Phase 3 live BCI)
- **Social playlists** — "music that makes both of us relaxed" from two users' brain data (blocks on Phase 6 multi-user auth)
- **Time-of-day patterns** — learn that user prefers calming music in the morning, energizing at noon (needs longitudinal per-user history)

## Platform

- **API access** — REST API for third-party integrations (other music apps, research tools)
- **Model marketplace** — share pre-trained models via braindecode's `push_to_hub()` / `from_pretrained()` HuggingFace integration
- **Research mode** — export data in BIDS format for academic use
