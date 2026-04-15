"""Pretrained EEG model wrapper with dual arousal/valence heads.

Wraps braindecode's CBraMod (pretrained on TUEG) with custom dual classification
heads for emotion recognition. Operates on raw EEG signals instead of hand-crafted
features, learning representations from the pretrained encoder.

Input: (batch, n_channels, n_times) raw EEG at 200Hz
Outputs:
    - arousal_logits: (batch, 2) binary classification
    - valence_logits: (batch, 2) binary classification
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PretrainedDualHead(nn.Module):
    """Dual-head emotion classifier wrapping a pretrained EEG encoder.

    Uses the pretrained model as a frozen/unfrozen feature extractor, then
    routes embeddings through separate arousal and valence classification heads.
    """

    def __init__(
        self,
        pretrained_model: nn.Module,
        embed_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = pretrained_model

        self.arousal_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.valence_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: raw EEG -> encoder -> dual heads.

        Args:
            x: Raw EEG tensor (batch, n_channels, n_times).

        Returns:
            (arousal_logits, valence_logits) each of shape (batch, 2).
        """
        out = self.encoder(x, return_features=True)
        features = out["features"]  # (batch, ..., embed_dim)
        # Flatten all spatial dims and average-pool to (batch, embed_dim)
        pooled = features.reshape(features.shape[0], -1, features.shape[-1]).mean(dim=1)

        arousal_logits = self.arousal_head(pooled)
        valence_logits = self.valence_head(pooled)
        return arousal_logits, valence_logits

    def freeze_backbone(self) -> None:
        """Freeze encoder parameters — only train classification heads."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen — training heads only")

    def unfreeze_backbone(self) -> None:
        """Unfreeze encoder for end-to-end fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen — full fine-tuning enabled")


def load_pretrained_dual_head(
    model_name: str = "cbramod",
    n_chans: int = 32,
    n_times: int = 800,
    sfreq: float = 200.0,
) -> PretrainedDualHead:
    """Load a pretrained model and wrap it with dual emotion heads.

    Args:
        model_name: Pretrained model identifier. Currently supports "cbramod".
        n_chans: Number of EEG channels in the input data.
        n_times: Number of time samples per segment.
        sfreq: Sampling frequency of the input data (Hz).
    """
    if model_name != "cbramod":
        msg = f"Unsupported pretrained model: {model_name}. Currently only 'cbramod' is supported."
        raise ValueError(msg)

    from braindecode.models import CBraMod

    backbone = CBraMod.from_pretrained(
        "braindecode/cbramod-pretrained",
        n_chans=n_chans,
        n_times=n_times,
        sfreq=sfreq,
        n_outputs=1,  # placeholder — we replace the head
    )

    # Determine embedding dimension by running a dummy forward pass
    backbone.eval()
    with torch.no_grad():
        dummy = torch.randn(1, n_chans, n_times)
        out = backbone(dummy, return_features=True)
        embed_dim = out["features"].shape[-1]

    logger.info(f"Loaded CBraMod pretrained encoder: {n_chans}ch, {n_times} samples @ {sfreq}Hz, embed_dim={embed_dim}")

    model = PretrainedDualHead(backbone, embed_dim=embed_dim)
    model.freeze_backbone()
    return model
