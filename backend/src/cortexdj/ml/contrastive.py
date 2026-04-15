"""EEG↔CLAP contrastive embedding model.

Joint 512-d embedding of 4-second EEG windows (via the CBraMod pretrained
backbone) and full tracks (via LAION-CLAP's audio encoder) such that an EEG
session can retrieve its nearest-neighbor tracks from a precomputed audio
index at inference time.

The loss handles the multi-subject duplicate-audio regime: in a single batch,
multiple EEG windows frequently target the same track (different subjects
watching the same DEAP stimulus, or different windows of the same trial).
Vanilla CLIP-style InfoNCE with `labels = arange(batch)` would treat those
as negatives-of-each-other; we use a soft-target formulation keyed on the
trial id instead, which reduces to vanilla InfoNCE when trial_ids are unique.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

CLAP_MODEL_ID = "laion/clap-htsat-unfused"
CLAP_TARGET_SFREQ = 48_000
EMBEDDING_DIM = 512

CBRAMOD_N_CHANS = 32
CBRAMOD_N_TIMES = 800
CBRAMOD_SFREQ = 200.0


def _load_cbramod_backbone() -> tuple[nn.Module, int]:
    """Load the CBraMod pretrained encoder for contrastive fine-tuning.

    Instantiated directly (not via pretrained.py::load_pretrained_dual_head)
    because the latter auto-freezes, logs a misleading "training heads only"
    message, and wraps the encoder in an unused dual-head module. We want
    the raw backbone with all parameters trainable — EEG foundation models
    collapse when frozen (EEG-FM-Bench 2025, arxiv 2508.17742).
    """
    from braindecode.models import CBraMod

    backbone = CBraMod.from_pretrained(
        "braindecode/cbramod-pretrained",
        n_chans=CBRAMOD_N_CHANS,
        n_times=CBRAMOD_N_TIMES,
        sfreq=CBRAMOD_SFREQ,
        n_outputs=1,  # placeholder — we never use the head
    )
    backbone.eval()
    with torch.no_grad():
        dummy = torch.randn(1, CBRAMOD_N_CHANS, CBRAMOD_N_TIMES)
        out = backbone(dummy, return_features=True)
        embed_dim = int(out["features"].shape[-1])
    backbone.train()
    for p in backbone.parameters():
        p.requires_grad = True
    logger.info(f"Loaded CBraMod pretrained encoder (trainable): embed_dim={embed_dim}")
    return backbone, embed_dim


class EegCLAPEncoder(nn.Module):
    """CBraMod backbone + SimCLR-style MLP projection → 512-d L2-normalized embedding.

    Projection head follows the canonical SimCLR / SupCon recipe:
    Linear → BatchNorm → nonlinearity → Dropout → Linear. BatchNorm between the
    two linears is empirically helpful for contrastive learning even when the
    backbone uses no BN. Dropout regularizes the projection without touching
    the backbone.
    """

    def __init__(self, *, projection_dim: int = EMBEDDING_DIM, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone, backbone_embed_dim = _load_cbramod_backbone()
        self.projection = nn.Sequential(
            nn.Linear(backbone_embed_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take (B, 32, 800) raw EEG at 200Hz → (B, 512) unit vectors."""
        out = self.backbone(x, return_features=True)
        features = out["features"]  # (B, ..., backbone_embed_dim)
        pooled = features.reshape(features.shape[0], -1, features.shape[-1]).mean(dim=1)
        projected = self.projection(pooled)
        return F.normalize(projected, dim=-1)

    def backbone_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def projection_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.projection.parameters() if p.requires_grad]


class ClapAudioEncoder:
    """Thin wrapper around LAION-CLAP's audio encoder for batch embedding.

    Not an nn.Module because CLAP is frozen and only used offline to populate
    the audio cache — we never backprop through it, so wrapping it as a
    callable avoids accidentally training it.
    """

    def __init__(self, device: torch.device, model_id: str = CLAP_MODEL_ID) -> None:
        from transformers import ClapModel, ClapProcessor

        self.device = device
        self.model_id = model_id
        model: Any = ClapModel.from_pretrained(model_id)
        self.model = model.to(device).eval()
        self.processor = ClapProcessor.from_pretrained(model_id)
        logger.info(f"Loaded CLAP audio encoder: {model_id}")

    @torch.no_grad()
    def embed_waveforms(self, waveforms: list[np.ndarray]) -> np.ndarray:
        """Embed a list of mono float32 waveforms at 48kHz → (N, 512) L2-normalized.

        transformers' `get_audio_features` returns a `BaseModelOutputWithPooling`
        whose `pooler_output` is the audio_projection output already L2-normalized
        per-row. We just unwrap and detach.
        """
        inputs = self.processor(
            audio=waveforms,
            sampling_rate=CLAP_TARGET_SFREQ,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output = self.model.get_audio_features(**inputs)
        features = output.pooler_output  # (N, 512), already L2-normalized
        result: np.ndarray = features.cpu().float().numpy()
        return result


def load_audio_waveform(m4a_path: Path) -> np.ndarray:
    """Decode an iTunes m4a preview to a mono 48kHz float32 waveform.

    CLAP's processor expects 48kHz mono. librosa handles m4a via audioread
    on systems where ffmpeg is installed — which includes macOS with Xcode
    or homebrew. If this fails on a bare system, install ffmpeg:
    `brew install ffmpeg`.
    """
    import librosa

    waveform, _ = librosa.load(str(m4a_path), sr=CLAP_TARGET_SFREQ, mono=True)
    return waveform.astype(np.float32)


def symmetric_info_nce(
    eeg_emb: torch.Tensor,
    audio_emb: torch.Tensor,
    trial_ids: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Soft-target symmetric InfoNCE.

    `eeg_emb` and `audio_emb` are L2-normalized `(B, D)` tensors produced by
    `EegCLAPEncoder` and `ClapAudioEncoder` respectively — `audio_emb[i]` is
    the CLAP embedding of whatever track `eeg_emb[i]` was recorded against.

    `trial_ids` is a `(B,)` int tensor encoding which DEAP trial each row
    came from. Rows sharing a trial_id are mutual positives: for EEG row i,
    the positive targets are every audio row j where trial_ids[j]==trial_ids[i],
    distributed uniformly. Identical audio vectors are expected.

    Reduces to vanilla CLIP InfoNCE when all trial_ids are unique.
    """
    temperature = temperature.clamp(min=1e-4)
    sim = (eeg_emb @ audio_emb.T) / temperature
    positive_mask = (trial_ids.unsqueeze(0) == trial_ids.unsqueeze(1)).float()
    target_e2a = positive_mask / positive_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    log_probs_e2a = F.log_softmax(sim, dim=1)
    loss_e2a = -(target_e2a * log_probs_e2a).sum(dim=1).mean()

    # Symmetric direction: audio → EEG. Since positive_mask is symmetric,
    # the transposed target distribution is just positive_mask.T renormalized.
    sim_a2e = sim.T
    target_a2e = positive_mask.T / positive_mask.T.sum(dim=1, keepdim=True).clamp_min(1.0)
    log_probs_a2e = F.log_softmax(sim_a2e, dim=1)
    loss_a2e = -(target_a2e * log_probs_a2e).sum(dim=1).mean()

    return (loss_e2a + loss_a2e) / 2


def retrieval_metrics(
    eeg_emb: torch.Tensor,
    audio_emb: torch.Tensor,
    trial_ids: torch.Tensor,
    *,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute top-k retrieval accuracy + mean reciprocal rank.

    For each EEG row, rank all unique audio targets in the batch by cosine
    similarity. A "hit@k" means the true target trial_id is in the top k.
    MRR is computed against the same unique-audio ranking.
    """
    unique_trial_ids, inverse = torch.unique(trial_ids, return_inverse=True)
    first_row_per_trial = torch.full((unique_trial_ids.numel(),), -1, dtype=torch.long, device=trial_ids.device)
    for row_idx in range(trial_ids.numel()):
        tid = int(inverse[row_idx].item())
        if first_row_per_trial[tid] < 0:
            first_row_per_trial[tid] = row_idx
    unique_audio = audio_emb[first_row_per_trial]

    sim = eeg_emb @ unique_audio.T  # (B, K)
    sorted_idx = sim.argsort(dim=1, descending=True)  # (B, K)
    target = inverse.unsqueeze(1)  # (B, 1)
    rank_of_positive = (sorted_idx == target).float().argmax(dim=1).float() + 1.0

    metrics: dict[str, float] = {}
    for k in k_values:
        hit = (rank_of_positive <= k).float().mean().item()
        metrics[f"top{k}"] = float(hit)
    metrics["mrr"] = float((1.0 / rank_of_positive).mean().item())
    return metrics


@torch.no_grad()
def encode_session(model: EegCLAPEncoder, segments: np.ndarray, device: torch.device) -> np.ndarray:
    """Aggregate a session's EEG windows into a single 512-d query vector.

    `segments` is a `(n_segments, 32, 800)` float32 array of already-resampled
    4-second EEG windows. Returns an L2-normalized numpy vector suitable for
    pgvector cosine similarity search.
    """
    model.eval()
    tensor = torch.from_numpy(segments.astype(np.float32)).to(device)
    embeddings = model(tensor)  # (N, 512) — already L2-normalized per-row
    mean = embeddings.mean(dim=0)
    pooled = F.normalize(mean, dim=0)
    return pooled.cpu().numpy()
