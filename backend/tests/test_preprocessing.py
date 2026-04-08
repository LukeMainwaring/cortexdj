"""Tests for EEG preprocessing: bandpass filtering, band powers, feature extraction."""

import numpy as np

from cortexdj.ml.preprocessing import (
    DEFAULT_SAMPLING_RATE,
    FREQ_BANDS,
    bandpass_filter,
    compute_band_powers,
    compute_differential_entropy,
    extract_features,
)


def _make_sine_signal(
    freq: float,
    duration: float = 4.0,
    fs: int = DEFAULT_SAMPLING_RATE,
    n_channels: int = 1,
) -> np.ndarray:
    """Generate a multi-channel sine wave at the given frequency."""
    t = np.arange(int(fs * duration)) / fs
    wave = np.sin(2 * np.pi * freq * t)
    return np.tile(wave, (n_channels, 1))


class TestBandpassFilter:
    def test_preserves_in_band_signal(self) -> None:
        signal_10hz = _make_sine_signal(10.0)
        filtered = bandpass_filter(signal_10hz, low=8.0, high=14.0)
        # In-band signal should retain most of its power
        original_power = float(np.mean(signal_10hz**2))
        filtered_power = float(np.mean(filtered**2))
        assert filtered_power / original_power > 0.8

    def test_attenuates_out_of_band_signal(self) -> None:
        signal_30hz = _make_sine_signal(30.0)
        filtered = bandpass_filter(signal_30hz, low=1.0, high=4.0)
        # Out-of-band signal should lose most power
        original_power = float(np.mean(signal_30hz**2))
        filtered_power = float(np.mean(filtered**2))
        assert filtered_power / original_power < 0.1

    def test_output_shape_matches_input(self) -> None:
        data = _make_sine_signal(10.0, n_channels=32)
        filtered = bandpass_filter(data, low=8.0, high=14.0)
        assert filtered.shape == data.shape


class TestComputeBandPowers:
    def test_returns_all_five_bands(self) -> None:
        data = _make_sine_signal(10.0, n_channels=32)
        powers = compute_band_powers(data)
        assert set(powers.keys()) == set(FREQ_BANDS.keys())

    def test_alpha_dominates_for_10hz_signal(self) -> None:
        data = _make_sine_signal(10.0, n_channels=1)
        powers = compute_band_powers(data)
        # 10 Hz falls in alpha band (8-14 Hz) — should be the strongest
        assert powers["alpha"] > powers["delta"]
        assert powers["alpha"] > powers["theta"]
        assert powers["alpha"] > powers["beta"]

    def test_all_powers_non_negative(self) -> None:
        data = np.random.randn(32, 512)
        powers = compute_band_powers(data)
        for power in powers.values():
            assert power >= 0.0


class TestComputeDifferentialEntropy:
    def test_returns_all_five_bands(self) -> None:
        data = np.random.randn(32, 512)
        de = compute_differential_entropy(data)
        assert set(de.keys()) == set(FREQ_BANDS.keys())

    def test_each_band_has_per_channel_values(self) -> None:
        n_channels = 32
        data = np.random.randn(n_channels, 512)
        de = compute_differential_entropy(data)
        for band_values in de.values():
            assert band_values.shape == (n_channels,)


class TestExtractFeatures:
    def test_output_shape_is_channels_times_bands(self) -> None:
        n_channels = 32
        data = np.random.randn(n_channels, 512)
        features = extract_features(data)
        expected_length = n_channels * len(FREQ_BANDS)
        assert features.shape == (expected_length,)

    def test_output_is_one_dimensional(self) -> None:
        data = np.random.randn(32, 512)
        features = extract_features(data)
        assert features.ndim == 1

    def test_deterministic_for_same_input(self) -> None:
        data = np.random.randn(32, 512)
        features_1 = extract_features(data)
        features_2 = extract_features(data)
        np.testing.assert_array_equal(features_1, features_2)
