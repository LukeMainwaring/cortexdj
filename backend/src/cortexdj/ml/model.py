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
    """Dual-head classifier for EEG emotion classification.

    Takes differential entropy features (n_channels * n_bands) and predicts
    binary arousal and valence labels.
    """

    def __init__(
        self,
        n_features: int = FEATURE_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Reshape features to (batch, 1, n_channels, n_bands) for conv processing
        self.n_channels = NUM_CHANNELS
        self.n_bands = NUM_BANDS

        # Spatial convolution across channels
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(NUM_CHANNELS, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # Temporal convolution across bands
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, NUM_BANDS)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Dual classification heads
        self.arousal_head = nn.Linear(hidden_dim, 2)
        self.valence_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (arousal_logits, valence_logits).

        Args:
            x: Feature tensor of shape (batch, n_features) or (batch, n_channels, n_bands).
        """
        if x.dim() == 2:
            # Reshape flat features to (batch, 1, n_channels, n_bands)
            x = x.view(-1, 1, self.n_channels, self.n_bands)

        # Conv path
        x = self.spatial_conv(x)  # (batch, 16, 1, n_bands)
        x = self.temporal_conv(x)  # (batch, 32, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 32)

        # Shared backbone
        x = self.backbone(x)

        # Dual heads
        arousal_logits = self.arousal_head(x)
        valence_logits = self.valence_head(x)

        return arousal_logits, valence_logits
