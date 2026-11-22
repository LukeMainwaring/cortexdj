# Research notes

Point-in-time research, where the `research` skill saves its findings. Each file is a dated snapshot of primary sources as read that day and is not maintained; the ADRs and rules hold the decisions it fed.

- `datasets-analysis.md`: the public EEG datasets weighed for CortexDJ beyond DEAP (emotion benchmarks, music-listening recordings, large pretraining corpora) and the three research paths they open, behind the dataset items in `docs/ROADMAP.md`.
- `pretrained-models-analysis.md`: braindecode's pretrained EEG model catalog weighed as backbones, and why CBraMod is the one in `ml/pretrained.py`.
- `2026-11-22-braindecode-mne-deap-pipeline.md`: loading, preprocessing and windowing DEAP's preprocessed files with braindecode 2.0 and MNE 1.13 (no maintained loader in the installed stack, what DEAP's own preprocessing already did, the small braindecode models and their input contract), checked against the installed packages and the local files.
- `vercel-chatbot-direction.md`: the state of Vercel's chat frontend lineage (`vercel/chatbot`, AI Elements, transports, the agent UI contract) in October 2026, behind the reference paragraph in `.agents/rules/frontend/vercel-ai-sdk.md`.
