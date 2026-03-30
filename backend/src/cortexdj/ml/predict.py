"""Inference wrapper for the trained EEGNet model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cortexdj.core.paths import CHECKPOINTS_DIR
from cortexdj.ml.dataset import scores_to_quadrant
from cortexdj.ml.model import EEGNetClassifier
from cortexdj.ml.preprocessing import compute_band_powers, extract_features

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "eegnet_best.pt"


@dataclass
class EEGPredictionResult:
    """Result from EEGNet inference on a single segment."""

    arousal_score: float
    valence_score: float
    arousal_class: str
    valence_class: str
    dominant_state: str
    band_powers: dict[str, float]


def load_model(checkpoint_path: str | Path | None = None) -> EEGNetClassifier:
    """Load a trained EEGNetClassifier from a checkpoint."""
    path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT

    if not path.exists():
        msg = f"Checkpoint not found: {path}. Run `uv run train-model` first."
        raise FileNotFoundError(msg)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = EEGNetClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(f"Loaded EEGNet from {path}")
    return model


def predict_segment(
    eeg_data: np.ndarray[tuple[int, ...], np.dtype[np.floating[object]]],
    model: EEGNetClassifier,
) -> EEGPredictionResult:
    """Run inference on a single EEG segment.

    Args:
        eeg_data: EEG signal array (n_channels x n_samples).
        model: Trained EEGNetClassifier.

    Returns:
        Prediction result with arousal/valence scores and dominant state.
    """
    features = extract_features(eeg_data)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        arousal_logits, valence_logits = model(features_tensor)

    arousal_probs = torch.softmax(arousal_logits[0], dim=0)
    valence_probs = torch.softmax(valence_logits[0], dim=0)

    arousal_score = float(arousal_probs[1].item())
    valence_score = float(valence_probs[1].item())

    arousal_class = "high" if arousal_score >= 0.5 else "low"
    valence_class = "high" if valence_score >= 0.5 else "low"

    # Map to emotion quadrant using raw score thresholds
    dominant_state = scores_to_quadrant(
        arousal_score * 10,  # Scale 0-1 back to 0-10 for quadrant mapping
        valence_score * 10,
    )

    band_powers = compute_band_powers(eeg_data)

    return EEGPredictionResult(
        arousal_score=round(arousal_score, 4),
        valence_score=round(valence_score, 4),
        arousal_class=arousal_class,
        valence_class=valence_class,
        dominant_state=dominant_state,
        band_powers=band_powers,
    )
