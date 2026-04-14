"""Unit tests for the EEG↔CLAP contrastive module.

Tests pin the empirical behavior of the loss, the retrieval metrics, and
the encoder's output shape/normalization. CBraMod and CLAP are mocked
to keep the suite fast and HF-free.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cortexdj.ml.contrastive import (
    EMBEDDING_DIM,
    retrieval_metrics,
    symmetric_info_nce,
)


class _FakeBackbone(nn.Module):
    """Minimal CBraMod-shaped backbone — (B, 32, 800) → features dict.

    Outputs `(B, T, 200)` so the caller's reshape+mean matches the real
    pretrained encoder's embedding layout.
    """

    def __init__(self, embed_dim: int = 200) -> None:
        super().__init__()
        self.linear = nn.Linear(32 * 800, 4 * embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, *, return_features: bool = False) -> dict[str, torch.Tensor]:
        flat = x.reshape(x.shape[0], -1)
        feat = self.linear(flat).reshape(x.shape[0], 4, self.embed_dim)
        return {"features": feat}


@pytest.fixture
def fake_encoder(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return an EegCLAPEncoder backed by a fake CBraMod.

    Has to be a fixture because the patch needs to happen before
    EegCLAPEncoder.__init__ calls _load_cbramod_backbone.
    """
    import cortexdj.ml.contrastive as mod

    monkeypatch.setattr(mod, "_load_cbramod_backbone", lambda: (_FakeBackbone(), 200))
    return mod.EegCLAPEncoder()


class TestSymmetricInfoNCE:
    def test_aligned_pairs_have_low_loss(self) -> None:
        torch.manual_seed(0)
        n = 8
        raw = torch.randn(n, EMBEDDING_DIM)
        emb = F.normalize(raw, dim=-1)
        trial_ids = torch.arange(n)
        # Scale up (low temperature) so logits separate sharply.
        temperature = torch.tensor(0.01)
        loss = symmetric_info_nce(emb, emb, trial_ids, temperature)
        assert loss.item() < 0.01

    def test_random_pairs_have_high_loss(self) -> None:
        torch.manual_seed(1)
        n = 8
        eeg = F.normalize(torch.randn(n, EMBEDDING_DIM), dim=-1)
        audio = F.normalize(torch.randn(n, EMBEDDING_DIM), dim=-1)
        trial_ids = torch.arange(n)
        temperature = torch.tensor(0.07)
        loss = symmetric_info_nce(eeg, audio, trial_ids, temperature)
        # Chance CE for 8 classes is log(8) ≈ 2.08 — random ≥ log(n) minus noise.
        assert loss.item() > 0.5

    def test_duplicate_trial_ids_strictly_better_than_vanilla_nce(self) -> None:
        # Two EEG rows from different windows/subjects of the SAME track
        # share a trial_id. With perfect alignment, the soft-target loss
        # hits its information-theoretic floor of log(|P|) ≈ log(2) ≈ 0.693
        # — NOT zero, because soft-target CE with |P|=2 distributes probability
        # mass across both positives, capping each log-prob at -log(2).
        # Crucially it's strictly better than vanilla InfoNCE's log(N) = log(4),
        # which would mis-label same-track rows as negatives.
        torch.manual_seed(2)
        audio_a = F.normalize(torch.randn(1, EMBEDDING_DIM), dim=-1)
        audio_b = F.normalize(torch.randn(1, EMBEDDING_DIM), dim=-1)
        audio = torch.cat([audio_a, audio_a, audio_b, audio_b], dim=0)
        eeg = audio.clone()
        trial_ids = torch.tensor([0, 0, 1, 1])
        loss = symmetric_info_nce(eeg, audio, trial_ids, torch.tensor(0.01))

        vanilla_floor = math.log(4)  # log(N) for 4 rows treated as 4 distinct classes
        soft_floor = math.log(2)  # log(|P|) for two positives per anchor

        assert loss.item() < vanilla_floor - 0.1
        assert math.isclose(loss.item(), soft_floor, abs_tol=0.05)

    def test_reduces_to_vanilla_info_nce_when_trial_ids_unique(self) -> None:
        torch.manual_seed(3)
        n = 6
        eeg = F.normalize(torch.randn(n, EMBEDDING_DIM), dim=-1)
        audio = F.normalize(torch.randn(n, EMBEDDING_DIM), dim=-1)
        trial_ids = torch.arange(n)
        temperature = torch.tensor(0.1)
        ours = symmetric_info_nce(eeg, audio, trial_ids, temperature).item()

        # Compute the vanilla symmetric CLIP loss for comparison.
        sim = (eeg @ audio.T) / temperature
        labels = torch.arange(n)
        expected = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))
        assert math.isclose(ours, float(expected.item()), rel_tol=1e-5)


class TestRetrievalMetrics:
    def test_top1_is_one_on_identity_matches(self) -> None:
        torch.manual_seed(4)
        n = 10
        emb = F.normalize(torch.randn(n, EMBEDDING_DIM), dim=-1)
        trial_ids = torch.arange(n)
        metrics = retrieval_metrics(emb, emb, trial_ids)
        assert metrics["top1"] == 1.0
        assert metrics["top5"] == 1.0
        assert metrics["mrr"] == 1.0

    def test_random_is_near_chance(self) -> None:
        torch.manual_seed(5)
        n = 32
        eeg = F.normalize(torch.randn(n, EMBEDDING_DIM), dim=-1)
        audio = F.normalize(torch.randn(n, EMBEDDING_DIM), dim=-1)
        trial_ids = torch.arange(n)
        metrics = retrieval_metrics(eeg, audio, trial_ids)
        # With 32 unique targets, random top1 ≈ 1/32 ≈ 0.03.
        assert metrics["top1"] < 0.25

    def test_duplicate_trial_ids_collapse_to_unique_pool(self) -> None:
        # 4 EEG rows, 2 unique trial_ids → retrieval pool size is 2.
        torch.manual_seed(6)
        audio_a = F.normalize(torch.randn(1, EMBEDDING_DIM), dim=-1)
        audio_b = F.normalize(torch.randn(1, EMBEDDING_DIM), dim=-1)
        audio = torch.cat([audio_a, audio_a, audio_b, audio_b], dim=0)
        eeg = audio.clone()
        trial_ids = torch.tensor([0, 0, 1, 1])
        metrics = retrieval_metrics(eeg, audio, trial_ids)
        assert metrics["top1"] == 1.0


class TestEegCLAPEncoder:
    def test_forward_shape_and_normalization(self, fake_encoder: Any) -> None:
        x = torch.randn(4, 32, 800)
        out = fake_encoder(x)
        assert out.shape == (4, EMBEDDING_DIM)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_backbone_and_projection_parameter_groups_disjoint(self, fake_encoder: Any) -> None:
        backbone_ids = {id(p) for p in fake_encoder.backbone_parameters()}
        projection_ids = {id(p) for p in fake_encoder.projection_parameters()}
        assert backbone_ids.isdisjoint(projection_ids)
        assert len(backbone_ids) > 0
        assert len(projection_ids) > 0


class TestSubjectSplit:
    def test_split_sizes_and_disjoint(self) -> None:
        from cortexdj.ml.contrastive_train import _split_subjects

        subjects = list(range(1, 33))
        train, val, test = _split_subjects(subjects, quick=False, seed=42)
        assert len(train) == 24
        assert len(val) == 4
        assert len(test) == 4
        assert set(train).isdisjoint(val)
        assert set(train).isdisjoint(test)
        assert set(val).isdisjoint(test)
        assert set(train) | set(val) | set(test) == set(subjects)

    def test_split_is_deterministic_for_same_seed(self) -> None:
        from cortexdj.ml.contrastive_train import _split_subjects

        subjects = list(range(1, 33))
        a = _split_subjects(subjects, quick=False, seed=42)
        b = _split_subjects(subjects, quick=False, seed=42)
        assert a == b

    def test_quick_split_sizes(self) -> None:
        from cortexdj.ml.contrastive_train import _split_subjects

        subjects = list(range(1, 33))
        train, val, test = _split_subjects(subjects, quick=True, seed=7)
        assert len(train) == 3
        assert len(val) == 1
        assert len(test) == 1
