# Pretrained EEG Models Reference

> braindecode 1.4.0 model catalog evaluated for CortexDJ. For current implementation details, see the codebase (`ml/pretrained.py`).

## Model Catalog

| Model | Params | Architecture | Channels | Sample Rate | Rationale for CortexDJ |
|-------|--------|-------------|----------|-------------|----------------------|
| CBraMod | 4.9M | Criss-cross transformer | Flexible | 200 Hz | Compact, ACPE supports 32ch→4ch (DEAP→Muse 2) transfer, ~1-epoch fine-tune. **Tier 1**. |
| REVE | 12M / 69M / 408M | Masked autoencoder | Flexible | Variable | Pretrained on 60K+ hrs / 92 datasets / 25K subjects; 4D Fourier PE enables cross-config transfer; linear probing works without heavy fine-tune. **Tier 1**. |
| LUNA | 7M / 43M / 311M | Conv+FFT + cross-attention + RoPE transformer | Flexible | Variable | Topology-invariant, linear in channel count. Muse 2 fallback if CBraMod's 32ch→4ch transfer degrades. 39.18% SEED-V vs. CBraMod 40.91%. **Tier 2**. |
| SignalJEPA | 3.5M | Conv encoder + Transformer | Flexible | Variable | Compact, strong embeddings. Useful for Phase 4 cross-session trend analysis. **Tier 2**. |
| EEGPT | 25.5M | Transformer + masking | 58 | 256 Hz | Capable but heavy; 58ch requirement forces DEAP interpolation. Deprioritized. |
| BENDR | 157M | Conv + Transformer (wav2vec-inspired) | 20 | 250 Hz | Best cross-subject generalization but prohibitive size for real-time BCI. Deprioritized. |
| BIOT | 3.2M | Linear attention transformer | 16–18 | Variable | Pretrained on sleep/epilepsy — wrong domain for emotion. Deprioritized. |
| LaBraM | 5.9M | Vision transformer | 128 (fixed) | Variable | Fixed 128ch montage — incompatible with DEAP/Muse 2. Deprioritized. |

## Recommendation

**Tier 1 (evaluate first):** CBraMod is the primary backbone — compact enough for real-time inference and the only flexible-channel option with a demonstrated single-epoch fine-tune on DEAP-scale data. REVE is the strongest secondary candidate given its breadth of pretraining and linear-probing efficacy.

**Tier 2 (evaluate if Tier 1 insufficient):** LUNA as a Muse 2 fallback if CBraMod's 4-channel transfer degrades badly; SignalJEPA for embedding-based trend analysis in Phase 4.

**Deprioritized:** EEGPT (channel count), BENDR (size), BIOT (wrong domain), LaBraM (fixed montage).

## API Patterns

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
# 1. Load pretrained, reset head
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
```

### Sharing Models

```python
# Push fine-tuned model to HuggingFace Hub
model.push_to_hub("username/cortexdj-emotion-cbramod")

# Load community model
model = CBraMod.from_pretrained("username/cortexdj-emotion-cbramod")
```

## Open Questions

1. **Sampling rate mismatch** — DEAP preprocessed data is 128 Hz; most pretrained models expect 200–256 Hz. Resampling is straightforward but may affect pretrained representations. Needs benchmark with and without resampling.

2. **Channel mapping** — DEAP uses 32 channels in a specific montage. Flexible-channel models (CBraMod, REVE, LUNA) handle this via positional encoding, but accuracy impact needs measurement.

3. **32ch→4ch transfer degradation** — CBraMod, REVE, and LUNA support arbitrary channel counts but none publish a 4-channel ablation. Needs direct measurement on DEAP with channels masked down to the Muse 2 montage (TP9/AF7/AF8/TP10).

4. **Real-time latency budget** — Phase 3 requires classification during live Spotify playback. CBraMod (4.9M) is the smallest transformer option. Needs CPU inference latency measurement for 4-second segments to confirm <500ms.

5. **Licensing** — Verify all pretrained model weights are licensed for commercial/open-source use. HuggingFace model cards should specify this.
