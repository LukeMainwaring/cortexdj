"""EEG preprocessing pipeline using MNE-Python and scipy.

Handles bandpass filtering, feature extraction (differential entropy across
5 frequency bands), and band power computation for EEG signals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import signal

# Standard EEG frequency bands (Hz)
FREQ_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 14.0),
    "beta": (14.0, 30.0),
    "gamma": (30.0, 40.0),
}

# DEAP preprocessed data sampling rate
DEFAULT_SAMPLING_RATE = 128


def bandpass_filter(
    data: npt.NDArray[np.floating[Any]],
    low: float,
    high: float,
    fs: int = DEFAULT_SAMPLING_RATE,
    order: int = 5,
) -> npt.NDArray[np.floating[Any]]:
    """Apply a Butterworth bandpass filter to EEG data.

    Args:
        data: EEG signal array (channels x samples) or (samples,).
        low: Low cutoff frequency (Hz).
        high: High cutoff frequency (Hz).
        fs: Sampling rate (Hz).
        order: Filter order.
    """
    nyquist = fs / 2.0
    b, a = signal.butter(order, [low / nyquist, high / nyquist], btype="band")
    filtered: npt.NDArray[np.floating[Any]] = signal.filtfilt(b, a, data, axis=-1)
    return filtered


def compute_differential_entropy(
    data: npt.NDArray[np.floating[Any]],
    fs: int = DEFAULT_SAMPLING_RATE,
) -> dict[str, npt.NDArray[np.floating[Any]]]:
    """Compute differential entropy (DE) features across frequency bands.

    DE for a Gaussian-distributed signal: DE = 0.5 * ln(2 * pi * e * variance)
    This is equivalent to 0.5 * ln(variance) + constant.

    Args:
        data: EEG signal array (channels x samples).
        fs: Sampling rate (Hz).

    Returns:
        Dict mapping band name to DE values per channel.
    """
    de_features: dict[str, npt.NDArray[np.floating[Any]]] = {}

    for band_name, (low, high) in FREQ_BANDS.items():
        filtered = bandpass_filter(data, low, high, fs)
        variance = np.var(filtered, axis=-1)
        # DE = 0.5 * ln(2 * pi * e * variance), simplified to 0.5 * ln(variance) + const
        de: npt.NDArray[np.floating[Any]] = 0.5 * np.log(variance + 1e-10)
        de_features[band_name] = de

    return de_features


def compute_band_powers(
    data: npt.NDArray[np.floating[Any]],
    fs: int = DEFAULT_SAMPLING_RATE,
) -> dict[str, float]:
    """Compute average power in each frequency band across all channels.

    Uses Welch's method for spectral estimation.

    Args:
        data: EEG signal array (channels x samples).
        fs: Sampling rate (Hz).

    Returns:
        Dict mapping band name to average power.
    """
    freqs, psd = signal.welch(data, fs=fs, nperseg=min(256, data.shape[-1]))

    band_powers: dict[str, float] = {}
    for band_name, (low, high) in FREQ_BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        # Average across frequency bins, then across channels
        power = float(np.mean(psd[..., mask]))
        band_powers[band_name] = round(power, 6)

    return band_powers


def extract_features(
    data: npt.NDArray[np.floating[Any]],
    fs: int = DEFAULT_SAMPLING_RATE,
) -> npt.NDArray[np.floating[Any]]:
    """Extract a flat feature vector from EEG data.

    Concatenates differential entropy across all bands and channels into
    a single feature vector: (n_channels * n_bands,).

    Args:
        data: EEG signal array (channels x samples).
        fs: Sampling rate (Hz).

    Returns:
        1D feature vector of shape (n_channels * 5,).
    """
    de = compute_differential_entropy(data, fs)
    # Stack: (n_bands, n_channels) -> flatten to (n_channels * n_bands,)
    feature_arrays = [de[band] for band in FREQ_BANDS]
    features: npt.NDArray[np.floating[Any]] = np.concatenate(feature_arrays)
    return features
