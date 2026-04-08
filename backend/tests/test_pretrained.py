"""Tests for the PretrainedDualHead model wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn

from cortexdj.ml.pretrained import PretrainedDualHead


def _make_mock_encoder(embed_dim: int = 64) -> nn.Module:
    """Create a minimal mock encoder that mimics braindecode's return_features API."""

    class MockEncoder(nn.Module):
        def __init__(self, embed_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(32, embed_dim)
            self._embed_dim = embed_dim

        def forward(self, x: torch.Tensor, return_features: bool = False) -> dict[str, torch.Tensor] | torch.Tensor:
            # x: (batch, n_chans, n_times)
            pooled = x.mean(dim=2)  # (batch, n_chans)
            features = self.linear(pooled).unsqueeze(1)  # (batch, 1, embed_dim)
            if return_features:
                result: dict[str, torch.Tensor] = {"features": features}
                return result
            squeezed: torch.Tensor = features.squeeze(1)
            return squeezed

    return MockEncoder(embed_dim)


class TestPretrainedDualHead:
    def test_forward_output_shapes(self) -> None:
        embed_dim = 64
        encoder = _make_mock_encoder(embed_dim)
        model = PretrainedDualHead(encoder, embed_dim=embed_dim)

        x = torch.randn(4, 32, 800)  # batch=4, 32 channels, 800 times
        arousal, valence = model(x)

        assert arousal.shape == (4, 2)
        assert valence.shape == (4, 2)

    def test_freeze_backbone(self) -> None:
        embed_dim = 64
        encoder = _make_mock_encoder(embed_dim)
        model = PretrainedDualHead(encoder, embed_dim=embed_dim)

        model.freeze_backbone()

        for param in model.encoder.parameters():
            assert not param.requires_grad
        for param in model.arousal_head.parameters():
            assert param.requires_grad
        for param in model.valence_head.parameters():
            assert param.requires_grad

    def test_unfreeze_backbone(self) -> None:
        embed_dim = 64
        encoder = _make_mock_encoder(embed_dim)
        model = PretrainedDualHead(encoder, embed_dim=embed_dim)

        model.freeze_backbone()
        model.unfreeze_backbone()

        for param in model.encoder.parameters():
            assert param.requires_grad

    def test_gradients_flow_through_heads_when_frozen(self) -> None:
        embed_dim = 64
        encoder = _make_mock_encoder(embed_dim)
        model = PretrainedDualHead(encoder, embed_dim=embed_dim)
        model.freeze_backbone()

        x = torch.randn(2, 32, 800)
        arousal, valence = model(x)
        loss = arousal.sum() + valence.sum()
        loss.backward()

        # Head params should have gradients
        for param in model.arousal_head.parameters():
            assert param.grad is not None

    def test_single_sample_batch(self) -> None:
        embed_dim = 64
        encoder = _make_mock_encoder(embed_dim)
        model = PretrainedDualHead(encoder, embed_dim=embed_dim)

        x = torch.randn(1, 32, 800)
        arousal, valence = model(x)

        assert arousal.shape == (1, 2)
        assert valence.shape == (1, 2)
