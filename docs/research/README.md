# Research notes

Point-in-time research, where the `research` skill saves its findings. Each file is a dated snapshot of primary sources as read that day and is not maintained; the ADRs and rules hold the decisions it fed.

- `datasets-analysis.md`: the public EEG datasets weighed for CortexDJ beyond DEAP (emotion benchmarks, music-listening recordings, large pretraining corpora) and the three research paths they open, behind the dataset items in `docs/ROADMAP.md`.
- `honest-model-comparison-on-small-samples.md`: how small-sample studies keep a reported score honest when a few models are compared on the same folds (nested cross-validation, held-out trials or participants, repeated cross-validation, paired tests across participants), sized for the 32-participant split of `docs/adr/0005-one-quadrant-classifier-per-participant.md`.
- `modal-fan-out-training.md`: Modal's setup for a small fan-out training run in November 2026 (image from the uv lockfile, data on a Volume, `map`, CPU against GPU, prices), behind `train-classifier-modal` in `cortexdj/training/modal_app.py`.
- `pretrained-models-analysis.md`: braindecode's pretrained EEG model catalog weighed as backbones, and why CBraMod is the contrastive encoder's backbone in `ml/contrastive.py`.
- `vercel-chatbot-direction.md`: the state of Vercel's chat frontend lineage (`vercel/chatbot`, AI Elements, transports, the agent UI contract) in October 2026, behind the reference paragraph in `.agents/rules/frontend/vercel-ai-sdk.md`.
