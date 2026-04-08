# Pretrained EEG Foundation Models: Analysis for CortexDJ

braindecode 1.4.0 ships 8 pretrained EEG foundation models with transfer learning APIs. This document evaluates their relevance to CortexDJ and outlines a migration path from the current custom pipeline.

## Executive Summary

CortexDJ supports two model backends: a custom `EEGNetClassifier` operating on hand-crafted differential entropy (DE) features, and a **CBraMod pretrained encoder** (shipped) with custom dual arousal/valence heads operating on raw EEG. The pretrained backend is implemented in `ml/pretrained.py` as `PretrainedDualHead`, selectable via `EEG_MODEL_BACKEND=cbramod`. This document evaluates the broader landscape of braindecode pretrained models for future expansion (e.g., REVE).

### Pipeline Comparison

```
Current Pipeline:
  Raw EEG (32ch, 128Hz, 4s)
    -> bandpass_filter() per band          [preprocessing.py]
    -> compute_differential_entropy()      [preprocessing.py]
    -> extract_features() -> (160,)        [preprocessing.py]
    -> EEGNetClassifier (dual-head)        [model.py]
    -> EEGPredictionResult                 [predict.py]

Pretrained Pipeline:
  Raw EEG (variable ch, variable Hz)
    -> resample if needed (most models expect 200-256 Hz)
    -> pretrained_model = Model.from_pretrained("hub-id")
    -> pretrained_model.reset_head(n_outputs=2)
    -> fine-tune (freeze encoder, train head)
    -> EEGPredictionResult                 [predict.py]
```

The pretrained approach eliminates manual feature engineering and leverages representations learned from thousands of hours of EEG data, which should generalize better — especially on small datasets like DEAP (32 participants).

## Model Catalog

| Model | Params | Architecture | Channels | Sample Rate | Training Corpus | Key Strength |
|-------|--------|-------------|----------|-------------|-----------------|-------------|
| EEGPT | 25.5M | Transformer + masking | 58 | 256 Hz | Multi-task SSL | Most flexible, general-purpose |
| CBraMod | 4.9M | Criss-cross transformer | Flexible | 200 Hz | TUEG (largest public EEG corpus) | Tiny, fast convergence (~1 epoch), any channel count |
| BENDR | 157M | Conv + Transformer (wav2vec-inspired) | 20 | 250 Hz | Contrastive predictive coding | Best cross-subject generalization |
| BIOT | 3.2M | Linear attention transformer | 16-18 | Variable | TUH Abnormal + SHHS | Sleep/epilepsy focus |
| Labram | 5.9M | Vision transformer | 128 (fixed) | Variable | SSL on large-scale EEG | Standard 10-20 montage only |
| REVE | 72-400M | Masked autoencoder | Flexible | Variable | 60K+ hours, 92 datasets, 25K subjects | Best for low-data, cross-config transfer |
| LUNA | Variable | CNN+FFT + cross-attention + RoPE Transformer | Flexible | Variable | SSL on EEG | Topology-invariant, channel-agnostic |
| SignalJEPA | 3.5M | Conv encoder + Transformer | Flexible (via chs_info) | Variable | Joint-embedding predictive | Compact, strong embeddings |

### Per-Model Notes

**EEGPT** — The most general-purpose option with 25.5M parameters. Uses spatio-temporal representation alignment and mask-based reconstruction for pretraining. Requires 58 channels at 256 Hz, which means channel interpolation would be needed for DEAP's 32 channels. Capable but heavy for real-time consumer use.

**CBraMod** — The strongest candidate for CortexDJ. At 4.9M parameters, it's compact enough for real-time inference. Its criss-cross transformer applies separate spatial and temporal attention with ~32% FLOP reduction. Critically, its Asymmetric Conditional Positional Encoding (ACPE) generalizes to **arbitrary channel counts** — meaning the same model can work with DEAP's 32 channels AND Muse 2's 4 channels. Achieves decent results after just 1 epoch of fine-tuning.

**BENDR** — At 157M parameters, this is the largest model. Inspired by wav2vec 2.0, it uses contrastive predictive coding and excels at cross-subject generalization. Too large for real-time BCI inference but excellent for offline batch analysis. Expects 20 channels at 250 Hz.

**BIOT** — Lightweight (3.2M) with linear attention for efficiency. However, it's pretrained on sleep staging and epilepsy detection data (TUH Abnormal, SHHS), making it less relevant for emotion recognition. Lower priority.

**Labram** — Uses a fixed 128-channel standard 10-20 montage. This is incompatible with both DEAP (32 channels) and Muse 2 (4 channels). Not viable for CortexDJ.

**REVE** — The most thoroughly pretrained model, trained on 60,000+ hours of EEG from 92 datasets and 25,000 subjects. Its 4D Fourier positional encoding enables cross-configuration transfer — meaning it can handle different electrode setups without retraining. Achieves state-of-the-art via linear probing alone (no heavy fine-tuning needed). The base variant (72M) is large but the linear probing capability means you may not need full fine-tuning.

**LUNA** — Topology-invariant architecture using CNN+FFT patch extraction, channel-unification via cross-attention, and RoPE Transformer. Handles varying channel counts by design. Available in base/large/huge sizes. Good fallback if CBraMod underperforms on variable-channel scenarios.

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
| Labram | Fixed 128-channel requirement is incompatible with all CortexDJ data sources |

## Pipeline Migration Architecture

### Dual-Head Strategy

The current `EEGNetClassifier` has dual output heads (arousal + valence) in a single model. braindecode's `reset_head(n_outputs)` creates a single classification head. Two approaches:

**Option A: Two separate pretrained models** — One fine-tuned for arousal, one for valence. Simple but doubles inference cost and memory.

**Option B: Custom dual-head wrapper** — Use `return_features=True` to extract the pretrained encoder's embeddings, then feed into a custom dual-head module. Preserves the single-model architecture:

```python
class PretrainedDualHead(nn.Module):
    def __init__(self, pretrained_model, embed_dim: int):
        super().__init__()
        self.encoder = pretrained_model
        self.arousal_head = nn.Linear(embed_dim, 2)
        self.valence_head = nn.Linear(embed_dim, 2)

    def forward(self, x):
        features = self.encoder(x, return_features=True)["features"]
        pooled = features.mean(dim=1)  # global average pooling
        return self.arousal_head(pooled), self.valence_head(pooled)
```

**Recommendation:** Option B — maintains the dual-head pattern, runs the encoder only once, and keeps `EEGPredictionResult` unchanged.

### Parallel Operation (Implemented)

Both pipelines are implemented and selectable via `EEG_MODEL_BACKEND` env var:

- `EEG_MODEL_BACKEND=eegnet` — DE feature pipeline (default)
- `EEG_MODEL_BACKEND=cbramod` — CBraMod pretrained pipeline
- Both produce the same `EEGPredictionResult` output
- `compute_band_powers()` from `preprocessing.py` remains useful for visualization regardless of model backend

### Files Implemented

| File | Status |
|------|--------|
| `ml/pretrained.py` | **New** — `PretrainedDualHead` wrapper with freeze/unfreeze, `load_pretrained_dual_head()` factory |
| `ml/predict.py` | **Updated** — `EEGModel` type alias, polymorphic `predict_segment()` and `load_model()` |
| `ml/train.py` | **Updated** — LOSO/grouped CV, model selection, two-phase pretrained training, `compare-models` CLI |
| `ml/dataset.py` | **Updated** — `DEAPFeatureDataset`, `DEAPRawDataset` (with 128→200Hz resampling), `load_dataset()` factory |
| `core/config.py` | **Updated** — `EEG_MODEL_BACKEND` setting |
| `app.py` | **Updated** — configurable model loading via `EEG_MODEL_BACKEND` |
| `agents/deps.py` | **Updated** — `EEGModel` type for `eeg_model` field |
| `preprocessing.py` | No changes — kept for band power visualization + backward compat |

## Roadmap Alignment

### Phase 2: Real EEG Datasets

Pretrained models are trained on massive EEG corpora (TUEG, 92 datasets for REVE). Fine-tuning on DEAP's 32 participants should significantly outperform training the custom EEGNet from scratch. Key actions:
- Benchmark CBraMod and REVE fine-tuned on DEAP vs. current EEGNetClassifier
- The pretrained models' built-in feature extraction eliminates the need to re-engineer DE features for each new dataset format

### Phase 3: Live BCI Device Integration

Muse 2 has only 4 EEG channels. The current 32-channel EEGNetClassifier **cannot work** with 4 channels without complete retraining and architecture changes (the spatial convolution kernel is `(32, 1)`). CBraMod and LUNA handle arbitrary channel counts natively — the same pretrained model can transfer from 32-channel training data to 4-channel inference. Key actions:
- Evaluate CBraMod inference latency on CPU for real-time classification (<500ms target)
- Test 32ch->4ch transfer accuracy degradation

### Phase 4: Advanced ML

Nearly every Phase 4 roadmap item maps to a pretrained model capability:

| Roadmap Item | Pretrained Solution |
|-------------|-------------------|
| CNN-Transformer hybrid | CBraMod, EEGPT, BENDR are production-ready transformer architectures |
| Transfer learning | `from_pretrained()` + `reset_head()` + fine-tune |
| Personalized models | Few-shot fine-tuning: freeze encoder, train head on ~10 min of user data |
| Attention visualization | Extract transformer attention weights for channel/timepoint importance maps |
| Cross-session trends | `return_features=True` embeddings enable trajectory analysis in embedding space |

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

1. **Sampling rate mismatch** — DEAP preprocessed data is 128 Hz; most pretrained models expect 200-256 Hz. Resampling with `mne.io.Raw.resample()` or `scipy.signal.resample` is straightforward but may affect pretrained representations. Need to benchmark with and without resampling.

2. **Channel mapping** — DEAP uses 32 channels in a specific montage. Pretrained models may expect different electrode positions. Models with flexible channel encoding (CBraMod, REVE, LUNA) handle this via positional encoding, but accuracy impact needs measurement.

3. **32ch to 4ch transfer degradation** — While CBraMod and LUNA support arbitrary channel counts, performance with only 4 channels (Muse 2) may degrade substantially. REVE shows 0.824 accuracy at 64 channels dropping to 0.660 at 1 channel. Need to benchmark at 4 channels specifically.

4. **Real-time latency budget** — Phase 3 requires classification during live Spotify playback. CBraMod (4.9M params) is the smallest transformer option. Need to measure CPU inference latency for 4-second segments to confirm <500ms is achievable.

5. **Dual-head fine-tuning dynamics** — Using a custom dual-head wrapper means the encoder is shared between arousal and valence tasks. Need to verify that multi-task fine-tuning doesn't degrade either head compared to single-task.

6. **Licensing** — Verify all pretrained model weights are licensed for commercial/open-source use. HuggingFace model cards should specify this.

---

*Analysis based on braindecode 1.4.0 (April 2026). Models and APIs may change in future versions.*
