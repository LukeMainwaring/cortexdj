# EEG Datasets Reference

> Public EEG dataset catalog evaluated for CortexDJ. Companion to [pretrained-models-analysis.md](pretrained-models-analysis.md). Sweep methodology and three research paths in the project's plan archive (`ultrathink-i-ve-been-reading-graceful-lollipop.md`).

## Headline findings

- **MOABB** (157 datasets, 3,517 subjects) is **emotion-empty**: motor imagery (61), P300/ERP (69), SSVEP (16), c-VEP (8), resting-state (3). No valence/arousal/music. Useful to CortexDJ only as generic SSL mass, which CBraMod/REVE/LUNA already absorb at larger scale.
- **Braindecode's** native `datasets` module wraps clinical / sleep / motor corpora (TUH, NMT, SleepPhysionet, BCI Comp IV4, HGD, SIENA, CHBMIT). **No emotion datasets.** Its value is the **tooling** — `from_pretrained()` / `reset_head()` / `BaseConcatDataset` + the channel-interpolation and Signal-JEPA fine-tune example notebooks.
- The high-leverage datasets live on **OpenNeuro, Stanford Digital Repository, NEMAR, and dataset-owner forms**. Three tiers below.

## Tier A — Emotion benchmarks beyond DEAP

Same valence/arousal label structure as DEAP; loader is mostly mechanical.

| Dataset | Subjects | Channels | Rate (Hz) | Stimuli | Labels | Access |
|---|---|---|---|---|---|---|
| **DEAP** (current) | 32 | 32 (+ peripheral) | 128 | 40 × 1-min music videos | V/A/D/liking Likert 1–9 | Academic form |
| **DREAMER** | 23 | 14 (Emotiv EPOC) | 128 | 18 movie clips | V/A/D Likert 1–5 | Free with form |
| **MAHNOB-HCI** | 27 | 32 (+ ECG, GSR, temp, eye gaze, face video) | 256 | 20 emotional videos | V/A/D + predictability + keywords | Academic form |
| **AMIGOS** | 40 | 14 | 128 | Short + long video clips, individual + group | V/A | Academic form |
| **SEED / SEED-IV / SEED-V** | 15 / 15 / 16 | 62 | 1000 | Film clips | Discrete pos/neu/neg → 4-class → 5-class | SJTU form |
| **FACED** | 123 | 28 | unspecified | Emotion-elicitation videos | 9 discrete emotion categories | Academic |
| **ASCERTAIN** | 58 | 8 | varies | Affective stimuli | Personality + V/A | Academic |

**Why this tier matters:** FACED is REVE's published emotion benchmark — adding it enables direct vs.-REVE comparison rather than inferring REVE-on-DEAP. MAHNOB-HCI's multimodal channels (GSR, ECG, eye gaze) open multimodal-fusion ablations. DREAMER's 14-channel Emotiv montage previews Phase 3's Muse 2 (4-channel) story before any hardware exists.

## Tier B — Music-listening datasets (EEG↔CLAP retrieval rescue)

EEG recorded during actual music perception, with raw audio distributed alongside. Directly addresses the failure mode documented in `docs/ROADMAP.md` "Deferred research: EEG↔CLAP" — DEAP's 1-min video clips encode mood/arousal, not track identity. Music-listening EEG should encode track-specific structure (beat, harmony, attention).

| Dataset | Subjects | Channels | Rate (Hz) | Stimuli | Labels | Audio? | Access |
|---|---|---|---|---|---|---|---|
| **MUSIN-G** (Pandey et al. 2022) | 20 | 128 (HGSN) | 250 | 12 full songs × 12 genres | 5-pt familiarity + 5-pt enjoyment | **Raw .wav alongside EEG**, BIDS | OpenNeuro `ds003774`, CC BY-NC-ND |
| **NMED-T** (Stanford, Losorelli et al.) | 20 | 125 (EGI) | unspecified | 10 commercial songs, duple meter | Tapping + ratings | Yes | Stanford Digital Repository |
| **NMED-H** (Stanford, sibling) | 48 | 125 | unspecified | Hindi pop songs | Familiarity + enjoyment | Yes | Stanford Digital Repository |
| **OpenMIIR** (Stober et al.) | 10 | 64 | 512 | 12 short fragments | Perception **and imagination** conditions | Yes | PDDL public domain |

**Why this tier matters:** today `contrastive_dataset.py` pairs each DEAP 4s EEG window with the CLAP embedding of a 30-second iTunes preview of the music video the subject watched — every link in that chain is weak. MUSIN-G replaces all of them at once: actual listening, full songs, raw audio in the same archive, plus a second supervision signal (familiarity/enjoyment) the InfoNCE loss doesn't currently use. NMED-T+H additionally share a common stimulus set across subjects, making cross-subject embedding alignment far easier than DEAP's 40 unique stimuli.

**Caveat:** combined sample size across all four is ~98 subjects, smaller than the DEAP-scale signal regime. Best treated as a 1–2 week timeboxed experiment, not a foundation.

## Tier C — Large pretraining corpora ("mini foundation model")

Re-pretraining a CBraMod-class model from scratch on TUH (~541k channel-hours) costs millions in compute — that's the public CBraMod / REVE checkpoint's job. The achievable version is an **intermediate stage** between the public backbone and DEAP fine-tuning.

| Corpus | Subjects | Channels | Rate | Tasks | Notes |
|---|---|---|---|---|---|
| **HBN-EEG** (Shirazi et al. 2024) | **3,000+** ages 5–21 | 128 (Geodesic) | 500 Hz | Resting state, **movie watching**, surround suppression, contrast change, sequence learning, symbol search | FAIR/BIDS, HED-annotated, 11 releases on NEMAR + OpenNeuro. Movie-watching is the closest published proxy for passive music-listening. Pretrain-on-Modal feasible. |
| **TUH-EEG / TUEG** | ~10,000 | varies | 250–500 | Clinical | Already inside CBraMod's pretraining. Touch only for ablations or alternate backbones. `braindecode.datasets.TUH`. |
| **NMT Scalp EEG** | ~700 | 19 | 200 | Clinical (Pakistan) | Mid-scale SSL corpus complementing TUH demographically. `braindecode.datasets.NMT`. |

**Why HBN-EEG is the standout:** 3,000-subject 128-channel movie-watching is the only freely accessible corpus of that scale on a passive-viewing task. Combined with the public CBraMod backbone, it enables a CortexDJ-flavored intermediate-pretraining stage at one to two orders of magnitude lower cost than TUEG-scale work, and yields a publishable open checkpoint as a side effect.

## Tier D — Tooling

| Tool | What it gives us |
|---|---|
| **[EEGain](https://github.com/EmotionLab/EEGain)** (EmotionLab 2025, CC BY 4.0, `pip install .`) | Unified loaders for **DEAP + SEED + SEED-IV + DREAMER + MAHNOB-HCI + AMIGOS** with standardized preprocessing (128 Hz, 4-s windows, 0.3–45 Hz bandpass, 50 Hz notch). Built-in models are EEGNet/TSception/DeepConvNet/ShallowConvNet — consume the dataset layer, plug CBraMod heads on top. Does **not** yet include FACED. |
| **braindecode `BaseConcatDataset` + `WindowsDataset`** | Cross-dataset concatenation at the windowing level. Pairs cleanly with EEGain. |
| **braindecode `plot_channel_interpolation` example** | Loads a pretrained foundation model on arbitrary channel sets — concrete API for the 32-ch ↔ 14-ch (DREAMER) ↔ 128-ch (MUSIN-G/HBN-EEG) montage problem. |
| **SPEED** (Scalable Preprocessing for EEG SSL, 2024) | Reference preprocessing pipeline if Tier-C pretraining is pursued. |

## Three research paths

Each path targets a different failure mode; they are largely independent.

### Path A — Cross-dataset emotion classifier

Train on DEAP + DREAMER + MAHNOB-HCI + FACED via EEGain; evaluate leave-one-dataset-out. Lowest risk, most-obviously-shippable improvement. Validates Phase 3's Muse 2 generalization before any hardware exists. Channel-count heterogeneity (14/28/32) handled by braindecode channel interpolation; label-space mismatch (Likert vs. 9-class discrete) handled by per-dataset label-mapping config.

### Path B — Music-listening retrieval rescue

Re-target the EEG↔CLAP contrastive pipeline from DEAP+iTunes-previews to MUSIN-G+NMED-T+NMED-H+OpenMIIR with full-song CLAP embeddings. The existing `EegCLAPEncoder` and InfoNCE loss in `ml/contrastive.py` drop in unchanged; loader and audio-pipeline rewrite is the work. Highest leverage if the hypothesis ("music-listening EEG encodes track identity better than video-watching EEG") holds. Best as a 1–2 week timeboxed experiment.

### Path C — HBN-EEG intermediate pretraining

SSL or self-distillation on HBN-EEG movie-watching between public CBraMod and DEAP fine-tuning. Highest ceiling, highest cost (tens to low hundreds of GPU-hours on Modal). Anchor on braindecode's relative-positioning example; data is hundreds of GB across 11 releases. Best deferred until Path A or B identifies a representation-quality bottleneck.

## Open questions

1. **EEGain coverage gap.** FACED is the dataset most worth adding for direct REVE comparison, but it's not in EEGain. Worth a custom loader, or wait for upstream?
2. **MUSIN-G channel mapping.** MUSIN-G is 128 channels (HGSN); CBraMod was fine-tuned on 32 (DEAP). Does the channel-interpolation path lose too much spatial detail, or should the contrastive encoder be retrained at native resolution?
3. **HBN-EEG age skew.** Subjects are 5–21 years old. Pretrained representations may transfer poorly to adult listeners — worth a small ablation against an adult-only subset (Release filtering) before committing to a full pretraining run.
4. **Licensing for music-listening retrieval.** MUSIN-G is CC BY-NC-ND. If a fine-tuned checkpoint encodes audio features from non-commercial material, downstream commercial use is murky. Worth a license review before any public release.

## References

- Braindecode dataset API: <https://braindecode.org/stable/api.html#datasets>
- MOABB dataset summary: <https://moabb.neurotechx.com/docs/dataset_summary.html>
- EEGain: <https://github.com/EmotionLab/EEGain>
- HBN-EEG paper: Shirazi et al. 2024, bioRxiv 10.1101/2024.10.03.615261
- MUSIN-G: OpenNeuro `ds003774`
- NMED-T: Stanford Digital Repository `purl.stanford.edu/jn859kj8079`
- OpenMIIR: <https://github.com/sstober/openmiir>
