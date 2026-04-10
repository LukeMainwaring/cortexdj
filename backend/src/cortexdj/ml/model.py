"""EEGNet-inspired dual-head classifier for EEG emotion recognition.

Architecture: Temporal convolution -> Depthwise spatial convolution ->
Separable convolution -> FC backbone -> dual heads (arousal + valence).

Based on Lawhern et al. (2018) "EEGNet: A Compact Convolutional Neural
Network for EEG-Based Brain-Computer Interfaces" but adapted for
differential entropy features rather than raw EEG.

Input: (batch, n_features) where n_features = n_channels * n_bands = 32 * 5 = 160
Outputs:
    - arousal_logits: (batch, 2) binary classification
    - valence_logits: (batch, 2) binary classification
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cortexdj.ml.preprocessing import FREQ_BANDS

NUM_CHANNELS = 32
NUM_BANDS = len(FREQ_BANDS)
FEATURE_DIM = NUM_CHANNELS * NUM_BANDS  # 160


class EEGNetClassifier(nn.Module):
    """Dual-head classifier taking differential entropy features (n_channels * n_bands)
    and predicting binary arousal and valence labels.
    """

    def __init__(
        self,
        n_features: int = FEATURE_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        spatial_filters: int = 32,
        temporal_filters: int = 64,
    ) -> None:
        super().__init__()

        self.n_channels = NUM_CHANNELS
        self.n_bands = NUM_BANDS

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, spatial_filters, kernel_size=(NUM_CHANNELS, 1)),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(spatial_filters, temporal_filters, kernel_size=(1, NUM_BANDS)),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.backbone = nn.Sequential(
            nn.Linear(temporal_filters, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.arousal_head = nn.Linear(hidden_dim, 2)
        self.valence_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (arousal_logits, valence_logits)."""
        if x.dim() == 2:
            x = x.view(-1, 1, self.n_channels, self.n_bands)

        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        x = x.view(x.size(0), -1)

        x = self.backbone(x)

        arousal_logits = self.arousal_head(x)
        valence_logits = self.valence_head(x)

        return arousal_logits, valence_logits
