"""Tests for the band features the quadrant classifier reads and the band powers a Segment shows."""

import numpy as np
import pytest

from cortexdj.ml.band_features import band_powers, differential_entropy
from cortexdj.ml.recipe import RECIPE

_FS = 128
_t = np.arange(4 * _FS) / _FS


def _window(*sines_hz: float, noise: float = 0.01) -> np.ndarray:
    """A 4-second, 32-channel window: the sum of unit sines at ``sines_hz`` plus a little noise."""
    rng = np.random.default_rng(0)
    wave = sum((np.sin(2 * np.pi * f * _t) for f in sines_hz), np.zeros_like(_t))
    return np.tile(wave, (32, 1)) + noise * rng.standard_normal((32, _t.size))


class TestDifferentialEntropy:
    def test_a_window_gives_128_features(self) -> None:
        assert differential_entropy(_window(10.0)).shape == (128,)

    def test_a_batch_of_windows_keeps_its_leading_axes(self) -> None:
        windows = np.stack([[_window(6.0), _window(20.0), _window(40.0)]] * 2)

        assert differential_entropy(windows).shape == (2, 3, 128)

    def test_the_feature_is_half_the_log_of_the_displayed_power(self) -> None:
        # A unit sine on every channel has variance 0.5, so every band's mean over channels is that band's power.
        window = _window(10.0, 20.0, noise=0.0)

        features = differential_entropy(window).reshape(32, 4)
        powers = band_powers(window)

        np.testing.assert_allclose(features.mean(axis=0), 0.5 * np.log(list(powers.values())), rtol=1e-4)

    def test_a_flat_channel_raises_rather_than_giving_minus_infinity(self) -> None:
        window = _window(10.0)
        window[5] = 0.0

        with pytest.raises(ValueError, match="non-finite"):
            differential_entropy(window)


class TestBandPowers:
    def test_the_four_bands_are_the_recipes_with_no_delta(self) -> None:
        assert list(band_powers(_window(10.0))) == ["theta", "alpha", "beta", "gamma"]
        assert [band.name for band in RECIPE.bands] == ["theta", "alpha", "beta", "gamma"]

    def test_a_sine_lands_in_its_own_band(self) -> None:
        for hz, band in [(6.0, "theta"), (10.5, "alpha"), (20.0, "beta"), (38.0, "gamma")]:
            powers = band_powers(_window(hz))

            assert max(powers, key=lambda name: powers[name]) == band
            assert powers[band] > 0.4  # most of the sine's 0.5 variance
