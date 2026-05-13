# Pretrained EEG Models Reference

> braindecode 1.4.0 model catalog evaluated for CortexDJ. For current implementation details, see the codebase (`ml/pretrained.py`).

## Model Catalog

| Model | Params | Architecture | Channels | Sample Rate | Rationale for CortexDJ |
|-------|--------|-------------|----------|-------------|----------------------|
| CBraMod | 4.9M | Criss-cross transformer | Flexible | 200 Hz | Compact, ACPE supports 32ch→4ch (DEAP→Muse 2) transfer, ~1-epoch fine-tune. **Tier 1**. |
| REVE | 12M / 69M / 408M | Masked autoencoder | Flexible | Variable | Pretrained on 60K+ hrs / 92 datasets / 25K subjects; 4D Fourier PE enables cross-config transfer; linear probing works without heavy fine-tune. **Tier 1**. |
| CodeBrain | unspecified in API doc | Decoupled TF tokenizer (VQ) + multi-scale EEGSSM (SSM + sliding-window attention) | Flexible | Flexible (default `patch_size=200` samples) | Pretrained on the TUH EEG Corpus (largest public dataset); 8 downstream tasks / 10 datasets in the ICLR 2026 paper; weights at `braindecode/codebrain-pretrained`; shares CBraMod's `(B, n_chans, n_times)` input contract — smallest-diff drop-in alternative. **Tier 1**. |
| LUNA | 7M / 43M / 311M | Conv+FFT + cross-attention + RoPE transformer | Flexible | Variable | Topology-invariant, linear in channel count. Muse 2 fallback if CBraMod's 32ch→4ch transfer degrades. 39.18% SEED-V vs. CBraMod 40.91%. **Tier 2**. |
| SignalJEPA | 3.5M | Conv encoder + Transformer | Flexible | Variable | Compact, strong embeddings. Useful for Phase 4 cross-session trend analysis. **Tier 2**. |
| TSception | tiny (defaults: 9 temporal + 6 spatial filters, 128 hidden) | Inception-style multi-window temporal + spatial convs | Flexible | Flexible | Designed for EEG emotion (Ding 2020), benchmarked on DEAP/AMIGOS/MAHNOB; HF entry `braindecode/TSception` exists but may be an architecture stub rather than a true pretrained checkpoint — confirm on first load. **Emotion baseline**. |
| DGCNN | small (Chebyshev order K=2) | Dynamic graph conv on electrode 3-D positions; learned adjacency + spectral filters | Flexible | Flexible | Canonical SEED emotion baseline (Song 2018); raw-EEG input in the braindecode implementation (paper used DE features); particularly relevant once Phase 2 SEED support lands. **Emotion baseline**. |
| EEGConformer | small–medium | Conv stem (temporal + spatial + pool) tokenizing into a lightweight transformer encoder | Flexible | Flexible (paper tuned at 250 Hz / 4 s) | Conv + attention without the pretraining cost; useful midpoint between EEGNet (DE features) and CBraMod (foundation). **Architecture baseline**. |
| EEGPT | 25.5M | Transformer + masking | 58 | 256 Hz | Capable but heavy; 58ch requirement forces DEAP interpolation. Deprioritized. |
| BENDR | 157M | Conv + Transformer (wav2vec-inspired) | 20 | 250 Hz | Best cross-subject generalization but prohibitive size for real-time BCI. Deprioritized. |
| BIOT | 3.2M | Linear attention transformer | 16–18 | Variable | Pretrained on sleep/epilepsy — wrong domain for emotion. Deprioritized. |
| LaBraM | 5.9M | Vision transformer | 128 (fixed) | Variable | Fixed 128ch montage — incompatible with DEAP/Muse 2. Deprioritized. |
| EEGMiner | small | Learnable generalized Gaussian filters + PLV / correlation connectivity features | Flexible | Flexible | Interpretability-first design (Ludwig 2024); no pretrained weights; better fit for the Phase 4 attention/feature-visualization line item than as a backbone. Deprioritized for backbone. |
| MEDFormer | medium | Multi-scale cross-channel patching + two-stage (intra- + inter-granularity) attention | Flexible | Flexible | General medical time-series (Wang 2024) with a sleep/ECG focus; no emotion benchmark, no pretrained weights. Deprioritized — domain mismatch. |
| Deep4Net | medium | 4-block conv (50 → 100 → 200 filters), pool + dropout, optional split first conv | Flexible | Flexible | Motor-imagery baseline (Schirrmeister 2017), from-scratch only; strictly inferior to keeping EEGNet as the small-from-scratch reference. Deprioritized. |

## Recommendation

**Tier 1 (evaluate first):** CBraMod is the current primary backbone — compact enough for real-time inference and the only flexible-channel option with a demonstrated single-epoch fine-tune on DEAP-scale data. REVE is the strongest secondary candidate given its breadth of pretraining and linear-probing efficacy. **CodeBrain** (ICLR 2026, [arxiv 2506.09110](https://arxiv.org/abs/2506.09110)) joins this tier as the smallest-diff CBraMod alternative — it shares the `(B, n_chans, n_times)` input contract that's already wired through `ml/pretrained.py` and `ml/contrastive.py`, and the existing dummy-forward-pass `embed_dim` inference would carry over unchanged.

**Tier 2 (evaluate if Tier 1 insufficient):** LUNA as a Muse 2 fallback if CBraMod's 4-channel transfer degrades badly; SignalJEPA for embedding-based trend analysis in Phase 4.

**Emotion-specific baselines (benchmark against, not replace):** TSception (Ding 2020) and DGCNN (Song 2018) are the canonical from-scratch architectures for EEG-emotion recognition — TSception built around inception-style temporal/spatial convs benchmarked on DEAP, DGCNN built around electrode-position graph convolution benchmarked on SEED. Including both in any model-zoo sweep tells us how much of the win comes from foundation-model pretraining vs the architecture itself. EEGConformer fills the same baseline role for "transformer without the pretraining cost." None replace CBraMod; all should appear as comparators in the eventual benchmark plot.

**Deprioritized:** EEGPT (channel count), BENDR (size), BIOT (wrong domain), LaBraM (fixed montage), MEDFormer (medical time-series, no emotion benchmark), Deep4Net (motor-imagery baseline, no pretrained weights). EEGMiner is deprioritized as a backbone but flagged for the Phase 4 attention-visualization line item — its learnable Gaussian filters and PLV connectivity features are a natural interpretability comparator to raw transformer attention.

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

2. **Channel mapping** — DEAP uses 32 channels in a specific montage. Flexible-channel models (CBraMod, REVE, LUNA) handle this via positional encoding, but accuracy impact needs measurement. Concrete API: braindecode's [`plot_channel_interpolation`](https://braindecode.org/stable/auto_examples/model_building/plot_channel_interpolation.html) example. Practical extremes to ablate against: DREAMER (14 ch, Emotiv) and MUSIN-G (128 ch, HGSN) — see [datasets-analysis.md](datasets-analysis.md).

3. **32ch→4ch transfer degradation** — CBraMod, REVE, and LUNA support arbitrary channel counts but none publish a 4-channel ablation. Needs direct measurement on DEAP with channels masked down to the Muse 2 montage (TP9/AF7/AF8/TP10).

4. **Real-time latency budget** — Phase 3 requires classification during live Spotify playback. CBraMod (4.9M) is the smallest transformer option. Needs CPU inference latency measurement for 4-second segments to confirm <500ms.

5. **Licensing** — Verify all pretrained model weights are licensed for commercial/open-source use. HuggingFace model cards should specify this.

6. **TSception / DGCNN HF entries — pretrained checkpoint or architecture stub?** The `braindecode/TSception` and `braindecode/DGCNN` HF repos are tagged "Feature Extraction" and were bulk-uploaded alongside dozens of other architecture-only entries, in contrast to the explicit `-pretrained` suffix on `braindecode/cbramod-pretrained` and `braindecode/codebrain-pretrained`. Confirm whether `from_pretrained()` actually loads non-random weights before treating these as pretrained models rather than from-scratch architectures.
