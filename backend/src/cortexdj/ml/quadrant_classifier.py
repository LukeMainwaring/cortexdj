"""The quadrant classifier: one linear layer on a window's 128 band features, scaling included."""

import torch
from torch import nn

# The model's two outputs, in column order: the logit of a high rating on each axis.
AXES = ("arousal", "valence")

SCORE_THRESHOLD = 0.5


def quadrant_of(arousal_score: float, valence_score: float) -> str:
    """The quadrant of two 0-to-1 scores, cut at 0.5 on each. The scores are the sigmoid as it comes, uncalibrated."""
    high_arousal, high_valence = arousal_score >= SCORE_THRESHOLD, valence_score >= SCORE_THRESHOLD
    if high_valence:
        return "excited" if high_arousal else "relaxed"
    return "stressed" if high_arousal else "calm"


class Standardize(nn.Module):
    """Scales features by the training fold's mean and standard deviation, held as buffers.

    Buffers, not parameters, so the optimizer and its weight decay never touch them and the
    saved state dict carries them: a loaded model needs no scaling applied from outside.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))

    def fit(self, inputs: torch.Tensor) -> None:
        """Set the scaling from ``inputs``, ``(n_windows, n_features)``: the training fold's windows only."""
        self.mean.copy_(inputs.mean(dim=0))
        self.std.copy_(inputs.std(dim=0).clamp_min(1e-8))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs - self.mean) / self.std


class QuadrantClassifier(nn.Module):
    """Unstandardized band features in, the arousal and valence logits out (``AXES`` order).

    The sigmoid of a logit is the stored score; the quadrant is cut at 0.5 on each.
    """

    def __init__(self, n_features: int = 128) -> None:
        super().__init__()
        self.standardize = Standardize(n_features)
        self.linear = nn.Linear(n_features, len(AXES))

    @classmethod
    def fitted_to(cls, inputs: torch.Tensor) -> QuadrantClassifier:
        """A fresh, untrained model whose scaling is set from the training fold's ``inputs``."""
        model = cls(inputs.shape[1])
        model.standardize.fit(inputs)
        return model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.linear(self.standardize(inputs))
        return logits
