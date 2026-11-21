"""Unit tests for the EEG↔CLAP contrastive encoder.

The encoder runs on a small fake backbone passed as `EegCLAPEncoder(backbone=...)`, so no test
loads CBraMod or CLAP.
"""

import pytest
import torch
import torch.nn as nn

from cortexdj.ml.contrastive import EegCLAPEncoder


class _FakeBackbone(nn.Module):
    """CBraMod-shaped backbone: (B, 32, 800) → {"features": (B, 4, 200)}, the real encoder's layout."""

    def __init__(self, embed_dim: int = 200) -> None:
        super().__init__()
        self.linear = nn.Linear(32 * 800, 4 * embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, *, return_features: bool = False) -> dict[str, torch.Tensor]:
        flat = x.reshape(x.shape[0], -1)
        feat = self.linear(flat).reshape(x.shape[0], 4, self.embed_dim)
        return {"features": feat}


@pytest.fixture
def encoder() -> EegCLAPEncoder:
    return EegCLAPEncoder(backbone=_FakeBackbone())


class TestEegCLAPEncoder:
    def test_forward_returns_unit_vectors_of_embedding_dim(self, encoder: EegCLAPEncoder) -> None:
        # The projection's BatchNorm needs a batch of at least 2 in train mode.
        out = encoder(torch.randn(4, 32, 800))
        assert out.shape == (4, 512)
        assert torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-5)

    def test_eval_mode_is_deterministic(self, encoder: EegCLAPEncoder) -> None:
        # In eval mode BatchNorm uses its running stats and Dropout is off; in train mode two
        # forwards on the same input would differ.
        encoder.eval()
        x = torch.randn(4, 32, 800)
        assert torch.allclose(encoder(x), encoder(x), atol=1e-6)
