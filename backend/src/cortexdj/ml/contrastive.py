"""EEG↔CLAP contrastive embedding model.

Joint 512-d embedding of 4-second EEG windows (via the CBraMod pretrained
backbone) and full tracks (via LAION-CLAP's audio encoder) such that an EEG
session can retrieve its nearest-neighbor tracks from a precomputed audio
index at inference time.
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


def _load_cbramod_backbone() -> nn.Module:
    """Load the CBraMod pretrained encoder for contrastive fine-tuning.

    The raw backbone with all parameters trainable: EEG foundation models
    collapse when frozen (EEG-FM-Bench 2025, arxiv 2508.17742), so do not
    freeze it or wrap it in classification heads.
    """
    from braindecode.models import CBraMod

    backbone: nn.Module = CBraMod.from_pretrained(
        "braindecode/cbramod-pretrained",
        n_chans=CBRAMOD_N_CHANS,
        n_times=CBRAMOD_N_TIMES,
        sfreq=CBRAMOD_SFREQ,
        n_outputs=1,  # placeholder — we never use the head
    )
    for p in backbone.parameters():
        p.requires_grad = True
    return backbone


def _measure_embed_dim(backbone: nn.Module) -> int:
    """Read the backbone's feature width from a dummy forward pass, leaving it in train mode."""
    backbone.eval()
    with torch.no_grad():
        dummy = torch.randn(1, CBRAMOD_N_CHANS, CBRAMOD_N_TIMES)
        out = backbone(dummy, return_features=True)
        embed_dim = int(out["features"].shape[-1])
    backbone.train()
    return embed_dim


class EegCLAPEncoder(nn.Module):
    """CBraMod backbone + SimCLR-style MLP projection → 512-d L2-normalized embedding.

    Projection head follows the canonical SimCLR / SupCon recipe:
    Linear → BatchNorm → nonlinearity → Dropout → Linear. BatchNorm between the
    two linears is empirically helpful for contrastive learning even when the
    backbone uses no BN. Dropout regularizes the projection without touching
    the backbone.

    `backbone` defaults to the pretrained CBraMod; a test passes a small module with the same
    `forward(x, return_features=True) -> {"features": ...}` contract instead. Do not rename the
    `backbone` or `projection` attributes: saved checkpoints' state-dict keys depend on them.
    """

    def __init__(
        self,
        *,
        projection_dim: int = EMBEDDING_DIM,
        dropout: float = 0.3,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone if backbone is not None else _load_cbramod_backbone()
        backbone_embed_dim = _measure_embed_dim(self.backbone)
        if backbone is None:
            logger.info(f"Loaded CBraMod pretrained encoder (trainable): embed_dim={backbone_embed_dim}")
        self.projection = nn.Sequential(
            nn.Linear(backbone_embed_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take (B, 32, 800) EEG at 200Hz, pre-scaled by `CBRAMOD_SCALE_FACTOR` → (B, 512) unit vectors."""
        out = self.backbone(x, return_features=True)
        features = out["features"]  # (B, ..., backbone_embed_dim)
        pooled = features.reshape(features.shape[0], -1, features.shape[-1]).mean(dim=1)
        projected = self.projection(pooled)
        return F.normalize(projected, dim=-1)


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

    CLAP's processor expects 48kHz mono. This does not work on the cached
    previews: under librosa 1.0.1 with soundfile 0.14.0, `librosa.load` raises
    `LibsndfileError: Format not recognised` on the AAC `.m4a` files, because
    libsndfile does not read AAC and librosa 1.0 removed the audioread
    fallback, so installing ffmpeg does not help. `seed-track-index` and
    contrastive training therefore cannot decode previews; the fix is tracked
    in `.scratch/audio-decoding/issues/01-decode-m4a-previews.md`.
    """
    import librosa

    waveform, _ = librosa.load(str(m4a_path), sr=CLAP_TARGET_SFREQ, mono=True)
    return waveform.astype(np.float32)


@torch.no_grad()
def encode_session(model: EegCLAPEncoder, segments: np.ndarray, device: torch.device) -> np.ndarray:
    """Aggregate a session's EEG windows into a single 512-d query vector.

    `segments` is a `(n_segments, 32, 800)` float32 array of 4-second EEG
    windows at 200Hz, pre-scaled to CBraMod's input range (callers should go
    through `trial_to_eeg_windows`, which applies `CBRAMOD_SCALE_FACTOR`).
    Returns an L2-normalized numpy vector suitable for pgvector cosine
    similarity search.
    """
    model.eval()
    tensor = torch.from_numpy(segments.astype(np.float32)).to(device)
    embeddings = model(tensor)  # (N, 512) — already L2-normalized per-row
    mean = embeddings.mean(dim=0)
    pooled = F.normalize(mean, dim=0)
    return pooled.cpu().numpy()
