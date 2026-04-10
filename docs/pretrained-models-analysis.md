# Pretrained EEG Models Reference

> braindecode 1.4.0 model catalog evaluated for CortexDJ. For current implementation details, see the codebase (`ml/pretrained.py`).

## Model Catalog

| Model | Params | Architecture | Channels | Sample Rate | Training Corpus | Key Strength |
|-------|--------|-------------|----------|-------------|-----------------|-------------|
| EEGPT | 25.5M | Transformer + masking | 58 | 256 Hz | Multi-task SSL | Most flexible, general-purpose |
| CBraMod | 4.9M | Criss-cross transformer | Flexible | 200 Hz | TUEG (largest public EEG corpus) | Tiny, fast convergence (~1 epoch), any channel count |
| BENDR | 157M | Conv + Transformer (wav2vec-inspired) | 20 | 250 Hz | Contrastive predictive coding | Best cross-subject generalization |
| BIOT | 3.2M | Linear attention transformer | 16-18 | Variable | TUH Abnormal + SHHS | Sleep/epilepsy focus |
| LaBraM | 5.9M | Vision transformer | 128 (fixed) | Variable | SSL on large-scale EEG | Standard 10-20 montage only |
| REVE | 12M / 69M / 408M | Masked autoencoder | Flexible | Variable | 60K+ hours, 92 datasets, 25K subjects | Best for low-data, cross-config transfer |
| LUNA | 7M / 43M / 311M | CNN+FFT + cross-attention + RoPE Transformer | Flexible | Variable | TUEG + Siena (~21K hours) | Topology-invariant, channel-agnostic |
| SignalJEPA | 3.5M | Conv encoder + Transformer | Flexible (via chs_info) | Variable | Joint-embedding predictive | Compact, strong embeddings |

## Per-Model Notes

**EEGPT** — The most general-purpose option with 25.5M parameters. Uses spatio-temporal representation alignment and mask-based reconstruction for pretraining. Requires 58 channels at 256 Hz, which means channel interpolation would be needed for DEAP's 32 channels. Capable but heavy for real-time consumer use.

**CBraMod** — The strongest candidate for CortexDJ. At 4.9M parameters, it's compact enough for real-time inference. Its criss-cross transformer applies separate spatial and temporal attention with ~32% FLOP reduction. Critically, its Asymmetric Conditional Positional Encoding (ACPE) generalizes to **arbitrary channel counts** — meaning the same model can work with DEAP's 32 channels AND Muse 2's 4 channels. Achieves decent results after just 1 epoch of fine-tuning.

**BENDR** — At 157M parameters, this is the largest model. Inspired by wav2vec 2.0, it uses contrastive predictive coding and excels at cross-subject generalization. Too large for real-time BCI inference but excellent for offline batch analysis. Expects 20 channels at 250 Hz.

**BIOT** — Lightweight (3.2M) with linear attention for efficiency. However, it's pretrained on sleep staging and epilepsy detection data (TUH Abnormal, SHHS), making it less relevant for emotion recognition. Lower priority.

**LaBraM** — Uses a fixed 128-channel standard 10-20 montage. This is incompatible with both DEAP (32 channels) and Muse 2 (4 channels). Not viable for CortexDJ.

**REVE** — The most thoroughly pretrained model, trained on 60,000+ hours of EEG from 92 datasets and 25,000 subjects. Its 4D Fourier positional encoding enables cross-configuration transfer — meaning it can handle different electrode setups without retraining. Achieves state-of-the-art via linear probing alone (no heavy fine-tuning needed). Available in Small (12M), Base (69M), and Large (408M) variants. Note: REVE's emotion-recognition benchmark is FACED, not DEAP — so any emotion-recognition numbers from the paper don't directly translate to CortexDJ's use case and would need to be measured on DEAP.

**LUNA** — Topology-invariant architecture using 1D conv + GroupNorm + GELU for temporal patches, fused with FFT (magnitude + phase) features, then channel-unification via learned-query cross-attention and a temporal transformer with RoPE. Linear (not quadratic) in channel count, ~300× FLOP reduction vs. naive attention. Available in Base (7M), Large (43M), and Huge (311M). Caveat for CortexDJ: LUNA's selling point is efficiency and topology-agnosticism, not raw emotion accuracy — it reports 39.18% on SEED-V vs. CBraMod's 40.91%. So CBraMod remains the stronger pretrained baseline for this project's valence/arousal use case; LUNA is only interesting as a Muse 2 fallback if CBraMod's 32ch→4ch transfer degrades badly.

**SignalJEPA** — Compact (3.5M) model using joint-embedding predictive architecture. Produces strong embedding representations useful for downstream analysis. Less studied for emotion classification specifically, but its embeddings could power cross-session trend analysis and similarity-based features.

## Recommended Models for CortexDJ

### Tier 1: Evaluate First

| Model | Why |
|-------|-----|
| **CBraMod** | Flexible channel count (32ch DEAP -> 4ch Muse 2), tiny (4.9M), fast convergence (~1 epoch fine-tuning), trained on TUEG. Covers Phase 2 and Phase 3 needs. |
| **REVE** | Best cross-dataset generalization (92 pretraining datasets), linear probing works without heavy fine-tuning, 4D positional encoding for cross-config transfer. Covers Phase 2 and Phase 4 needs. |

### Tier 2: Evaluate if Tier 1 Insufficient

| Model | Why |
|-------|-----|
| **LUNA** | Topology-invariant (different montages), channel-agnostic. Good fallback for Muse 2 integration if CBraMod struggles with 4 channels. |
| **SignalJEPA** | Compact (3.5M), strong embeddings. Useful for Phase 4 cross-session trend analysis via embedding trajectories. |

### Tier 3: Deprioritized

| Model | Why Deprioritized |
|-------|-------------------|
| EEGPT | Capable but 25.5M params and requires 58 channels — needs interpolation for 32ch data |
| BENDR | Excellent generalization but 157M params prohibitive for real-time BCI |
| BIOT | Wrong domain focus (sleep/epilepsy, not emotion recognition) |
| LaBraM | Fixed 128-channel requirement is incompatible with all CortexDJ data sources |

## Key API Patterns

### Loading a Pretrained Model

```python
from braindecode.models import CBraMod

model = CBraMod.from_pretrained(
    "braindecode/cbramod-pretrained",
    n_outputs=4,  # auto-rebuilds head if different from saved
)
```

### Adapting for a New Task

```python
model = CBraMod.from_pretrained("braindecode/cbramod-pretrained")
model.reset_head(n_outputs=2)  # binary arousal classification
```

### Extracting Embeddings

```python
x = torch.randn(1, n_chans, n_times)
out = model(x, return_features=True)
embeddings = out["features"]  # encoder representations
```

### Fine-Tuning Recipe

```python
# 1. Load pretrained
model = CBraMod.from_pretrained("braindecode/cbramod-pretrained")
model.reset_head(n_outputs=2)

# 2. Freeze encoder
for param in model.parameters():
    param.requires_grad = False
for param in model.final_layer.parameters():
    param.requires_grad = True

# 3. Train head on DEAP
optimizer = torch.optim.Adam(model.final_layer.parameters(), lr=1e-3)
# ... training loop ...

# 4. Optional: unfreeze encoder, fine-tune end-to-end with lower LR
for param in model.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# ... continued training ...
```

### Sharing Models

```python
# Push fine-tuned model to HuggingFace Hub
model.push_to_hub("username/cortexdj-emotion-cbramod")

# Load community model
model = CBraMod.from_pretrained("username/cortexdj-emotion-cbramod")
```

## Open Questions

1. **Sampling rate mismatch** — DEAP preprocessed data is 128 Hz; most pretrained models expect 200-256 Hz. Resampling is straightforward but may affect pretrained representations. Need to benchmark with and without resampling.

2. **Channel mapping** — DEAP uses 32 channels in a specific montage. Models with flexible channel encoding (CBraMod, REVE, LUNA) handle this via positional encoding, but accuracy impact needs measurement.

3. **32ch to 4ch transfer degradation** — While CBraMod and LUNA support arbitrary channel counts, performance with only 4 channels (Muse 2) may degrade substantially. None of CBraMod/REVE/LUNA publish a 4-channel ablation in their papers, so this needs to be measured directly on DEAP with channels masked down to the Muse 2 montage (TP9/AF7/AF8/TP10).

4. **Real-time latency budget** — Phase 3 requires classification during live Spotify playback. CBraMod (4.9M params) is the smallest transformer option. Need to measure CPU inference latency for 4-second segments to confirm <500ms is achievable.

5. **Licensing** — Verify all pretrained model weights are licensed for commercial/open-source use. HuggingFace model cards should specify this.
