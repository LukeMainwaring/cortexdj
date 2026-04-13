from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from scipy.signal import resample

from cortexdj.core.paths import CHECKPOINTS_DIR
from cortexdj.ml.dataset import CBRAMOD_SEGMENT_SAMPLES, scores_to_quadrant
from cortexdj.ml.model import EEGNetClassifier
from cortexdj.ml.preprocessing import compute_band_powers, extract_features
from cortexdj.ml.pretrained import PretrainedDualHead, load_pretrained_dual_head

logger = logging.getLogger(__name__)

type EEGModel = EEGNetClassifier | PretrainedDualHead

CHECKPOINT_PATHS: dict[str, Path] = {
    "eegnet": CHECKPOINTS_DIR / "eegnet_best.pt",
    "cbramod": CHECKPOINTS_DIR / "cbramod_best.pt",
}


@dataclass
class EEGPredictionResult:
    arousal_score: float
    valence_score: float
    arousal_class: str
    valence_class: str
    dominant_state: str
    band_powers: dict[str, float]


def load_model(
    checkpoint_path: str | Path | None = None,
    model_type: str | None = None,
) -> EEGModel:
    """Load a trained model from checkpoint."""
    if checkpoint_path:
        path = Path(checkpoint_path)
    elif model_type:
        path = CHECKPOINT_PATHS.get(model_type, CHECKPOINT_PATHS["eegnet"])
    else:
        path = CHECKPOINT_PATHS["eegnet"]

    if not path.exists():
        msg = f"Checkpoint not found: {path}. Run `uv run train-model` first."
        raise FileNotFoundError(msg)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    resolved_type = model_type or checkpoint["model_type"]

    # Lazy import to avoid a cycle: train imports `EEGModel` from this module.
    from cortexdj.ml.train import CHECKPOINT_SCHEMA_VERSION

    schema = checkpoint.get("schema_version")
    if not isinstance(schema, int) or schema < CHECKPOINT_SCHEMA_VERSION:
        msg = (
            f"Checkpoint at {path} has schema {schema!r} "
            f"(expected {CHECKPOINT_SCHEMA_VERSION}). Retrain with "
            f"`uv run train-model --model {resolved_type}` and try again."
        )
        raise RuntimeError(msg)

    if resolved_type == "cbramod":
        model: EEGModel = load_pretrained_dual_head()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info(f"Loaded CBraMod from {path}")
    else:
        model = EEGNetClassifier()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info(f"Loaded EEGNet from {path}")

    return model


def predict_segment(
    eeg_data: npt.NDArray[np.floating[Any]],
    model: EEGModel,
) -> EEGPredictionResult:
    """Run inference on a single EEG segment (n_channels x n_samples)."""
    if isinstance(model, PretrainedDualHead):
        # Raw EEG path: resample 128Hz -> 200Hz for CBraMod
        resampled = resample(eeg_data, CBRAMOD_SEGMENT_SAMPLES, axis=1)
        input_tensor = torch.tensor(resampled, dtype=torch.float32).unsqueeze(0)
    else:
        # DE features path for EEGNet
        features = extract_features(eeg_data)
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        arousal_logits, valence_logits = model(input_tensor)

    arousal_probs = torch.softmax(arousal_logits[0], dim=0)
    valence_probs = torch.softmax(valence_logits[0], dim=0)

    arousal_score = float(arousal_probs[1].item())
    valence_score = float(valence_probs[1].item())

    arousal_class = "high" if arousal_score >= 0.5 else "low"
    valence_class = "high" if valence_score >= 0.5 else "low"

    # Scale 0-1 back to 0-10 for quadrant mapping
    dominant_state = scores_to_quadrant(
        arousal_score * 10,
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
