# Research notes

Point-in-time research, where the `research` skill saves its findings. Each file is a dated snapshot of primary sources as read that day and is not maintained; the ADRs and rules hold the decisions it fed.

- `bigger-models-on-band-features.md`: what published trial-safe results say about MLPs, deep belief networks and graph convolutions on one window's band features against a simple classifier (DEAP first, then SEED and MAHNOB-HCI), the regularization they used, and which large gains come from window-level splits; behind the architecture bake-off item in `docs/ROADMAP.md`.
- `datasets-analysis.md`: the public EEG datasets weighed for CortexDJ beyond DEAP (emotion benchmarks, music-listening recordings, large pretraining corpora) and the three research paths they open, behind the dataset items in `docs/ROADMAP.md`.
- `modal-fan-out-training.md`: Modal's setup for a small fan-out training run in November 2026 (image from the uv lockfile, data on a Volume, `map`, CPU against GPU, prices), behind `train-classifier-modal` in `cortexdj/training/modal_app.py`.
- `pretrained-models-analysis.md`: braindecode's pretrained EEG model catalog weighed as backbones, and why CBraMod is the contrastive encoder's backbone in `ml/contrastive.py`.
- `vercel-chatbot-direction.md`: the state of Vercel's chat frontend lineage (`vercel/chatbot`, AI Elements, transports, the agent UI contract) in October 2026, behind the reference paragraph in `.agents/rules/frontend/vercel-ai-sdk.md`.
