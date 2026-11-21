"""Retrieval-slice fixtures for the integration tier: a real encoder on a tiny backbone, and a fake DEAP directory.

The encoder is a real ``EegCLAPEncoder`` (projection head, pooling, normalization) whose
CBraMod backbone is replaced by one linear layer, so no test downloads the pretrained
weights. Its weights are random: a test that needs a known query vector embeds the session
through the service itself and builds its index rows around that vector.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from cortexdj.features.retrieval.encoder import ContrastiveEncoder
from cortexdj.ml.contrastive import EegCLAPEncoder

# DEAP's preprocessed layout: 40 channels (32 EEG first), 384 baseline + 7680 stimulus samples at 128Hz.
_DEAP_CHANNELS = 40
_DEAP_SAMPLES = 8064


class FakeBackbone(nn.Module):
    """CBraMod's ``forward(x, return_features=True) -> {"features": (B, patches, dim)}`` contract, tiny."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(32 * 800, 8)

    def forward(self, x: torch.Tensor, *, return_features: bool = False) -> dict[str, torch.Tensor]:
        return {"features": self.linear(x.reshape(x.shape[0], -1)).reshape(x.shape[0], 1, 8)}


@pytest.fixture
def encoder() -> ContrastiveEncoder:
    torch.manual_seed(0)
    return ContrastiveEncoder(model=EegCLAPEncoder(backbone=FakeBackbone()), device=torch.device("cpu"))


@pytest.fixture
def deap_dir(tmp_path: Path) -> Path:
    """A DEAP directory holding one participant, ``s01.dat`` (session participant ``P01``), with two trials."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((2, _DEAP_CHANNELS, _DEAP_SAMPLES)).astype(np.float32)
    labels = rng.uniform(1, 9, (2, 4)).astype(np.float32)
    with (tmp_path / "s01.dat").open("wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)
    return tmp_path
