"""Torch device selection shared by inference and the offline embedding cache."""

import torch


def get_device() -> torch.device:
    """Select the best available device: CUDA → MPS (Apple Silicon) → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
