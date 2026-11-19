"""Unit tests for the EEG↔CLAP contrastive module: the loss, the retrieval metrics, and the encoder.

The encoder runs on a small fake backbone passed as `EegCLAPEncoder(backbone=...)`, so no test
loads CBraMod or CLAP.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cortexdj.ml.contrastive import (
    EMBEDDING_DIM,
    EegCLAPEncoder,
    retrieval_metrics,
    symmetric_info_nce,
)


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


class TestSymmetricInfoNCE:
    def test_aligned_pairs_have_low_loss(self) -> None:
        torch.manual_seed(0)
        emb = F.normalize(torch.randn(8, EMBEDDING_DIM), dim=-1)
        # A low temperature separates the logits sharply.
        loss = symmetric_info_nce(emb, emb, torch.arange(8), torch.tensor(0.01))
        assert loss.item() < 0.01

    def test_random_pairs_have_high_loss(self) -> None:
        torch.manual_seed(1)
        eeg = F.normalize(torch.randn(8, EMBEDDING_DIM), dim=-1)
        audio = F.normalize(torch.randn(8, EMBEDDING_DIM), dim=-1)
        loss = symmetric_info_nce(eeg, audio, torch.arange(8), torch.tensor(0.07))
        # Chance cross-entropy over 8 classes is log(8) ≈ 2.08.
        assert loss.item() > 0.5

    def test_duplicate_trial_ids_reach_the_soft_target_floor(self) -> None:
        # Two rows per track share a trial_id. With perfect alignment the soft-target loss
        # spreads probability over both positives, so its floor is log(2) ≈ 0.693, well under
        # the log(4) ≈ 1.386 that vanilla InfoNCE would reach by treating same-track rows as
        # negatives of each other.
        torch.manual_seed(2)
        audio_a = F.normalize(torch.randn(1, EMBEDDING_DIM), dim=-1)
        audio_b = F.normalize(torch.randn(1, EMBEDDING_DIM), dim=-1)
        audio = torch.cat([audio_a, audio_a, audio_b, audio_b], dim=0)
        loss = symmetric_info_nce(audio.clone(), audio, torch.tensor([0, 0, 1, 1]), torch.tensor(0.01))
        assert loss.item() == pytest.approx(0.693, abs=0.05)

    def test_unique_trial_ids_give_the_vanilla_info_nce_value(self) -> None:
        # Asymmetric on purpose, so the two directions and the temperature each matter. At
        # temperature 0.5 the logits are [[2, 1.2], [0, 1.6]]:
        #   EEG → audio: (log(1 + e^-0.8) + log(1 + e^-1.6)) / 2 ≈ 0.27750
        #   audio → EEG: (log(1 + e^-2.0) + log(1 + e^-0.4)) / 2 ≈ 0.31997
        # and plain symmetric cross-entropy is their mean.
        eeg = torch.eye(2)
        audio = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
        loss = symmetric_info_nce(eeg, audio, torch.tensor([0, 1]), torch.tensor(0.5))
        assert loss.item() == pytest.approx(0.29874, abs=1e-5)


class TestRetrievalMetrics:
    def test_identity_matches_score_perfectly(self) -> None:
        torch.manual_seed(4)
        emb = F.normalize(torch.randn(10, EMBEDDING_DIM), dim=-1)
        metrics = retrieval_metrics(emb, emb, torch.arange(10))
        assert metrics["top1"] == 1.0
        assert metrics["top5"] == 1.0
        assert metrics["mrr"] == 1.0

    def test_random_is_near_chance(self) -> None:
        torch.manual_seed(5)
        eeg = F.normalize(torch.randn(32, EMBEDDING_DIM), dim=-1)
        audio = F.normalize(torch.randn(32, EMBEDDING_DIM), dim=-1)
        metrics = retrieval_metrics(eeg, audio, torch.arange(32))
        # With 32 unique targets, chance top1 is 1/32 ≈ 0.03.
        assert metrics["top1"] < 0.25

    def test_duplicate_trial_ids_collapse_to_unique_pool(self) -> None:
        # 4 EEG rows over 2 unique trial_ids: the retrieval pool has 2 targets.
        torch.manual_seed(6)
        audio_a = F.normalize(torch.randn(1, EMBEDDING_DIM), dim=-1)
        audio_b = F.normalize(torch.randn(1, EMBEDDING_DIM), dim=-1)
        audio = torch.cat([audio_a, audio_a, audio_b, audio_b], dim=0)
        metrics = retrieval_metrics(audio.clone(), audio, torch.tensor([0, 0, 1, 1]))
        assert metrics["top1"] == 1.0


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

    def test_backbone_and_projection_parameter_groups_are_disjoint(self, encoder: EegCLAPEncoder) -> None:
        backbone_ids = {id(p) for p in encoder.backbone_parameters()}
        projection_ids = {id(p) for p in encoder.projection_parameters()}
        assert backbone_ids.isdisjoint(projection_ids)
        # The fake backbone's one Linear: weight + bias.
        assert len(backbone_ids) == 2
        # Two Linears and a BatchNorm, each weight + bias.
        assert len(projection_ids) == 6
