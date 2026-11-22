"""The band features the quadrant classifier reads, and the band powers a Segment shows.

Both come from one band-pass of the 4-second window, so the model reads the log of what the
chart draws. Training and seeding both call these functions; do not compute either a second way.
A caller that needs both computes ``band_variance`` once and derives each from it with
``differential_entropy_from`` and ``band_powers_from``, so no window is band-passed twice.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import signal

from cortexdj.ml.recipe import RECIPE, TrainingRecipe


def band_variance(windows: npt.NDArray[np.floating[Any]], recipe: TrainingRecipe = RECIPE) -> npt.NDArray[np.float64]:
    """Each channel's variance in each of the recipe's bands: ``(..., channels, samples)`` to ``(..., channels, bands)``.

    A zero-phase Butterworth band-pass in second-order sections (``sosfiltfilt``), which stays
    stable at these orders where the ``b, a`` form does not.
    """
    per_band = []
    for band in recipe.bands:
        sos = signal.butter(
            recipe.filter_order, [band.low, band.high], btype="band", fs=recipe.sample_rate, output="sos"
        )
        per_band.append(signal.sosfiltfilt(sos, windows, axis=-1).var(axis=-1))
    return np.stack(per_band, axis=-1)


def differential_entropy(
    windows: npt.NDArray[np.floating[Any]], recipe: TrainingRecipe = RECIPE
) -> npt.NDArray[np.float64]:
    """The classifier's input: half the log of each band variance, ``(..., 32, 512)`` to ``(..., 128)``."""
    return differential_entropy_from(band_variance(windows, recipe))


def differential_entropy_from(variance: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The classifier's input from ``band_variance``'s output: ``(..., channels, bands)`` to ``(..., channels * bands)``.

    Channel-major: feature ``4 * c + b`` is channel ``c`` in band ``b``. The Gaussian constant
    ``0.5 * log(2 pi e)`` is left out because the model standardizes its input. Raises
    ``ValueError`` on a non-finite feature (a flat channel's zero variance logs to ``-inf``, a
    NaN sample spreads), because the standardizing layer would turn it into NaN scores silently.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        de: npt.NDArray[np.float64] = 0.5 * np.log(variance)
    finite = np.isfinite(de).all(axis=-1)
    if not finite.all():
        bad = np.argwhere(~finite)
        raise ValueError(
            f"{len(bad)} channel window(s) give non-finite band features (zero or NaN band variance), "
            f"first at index {tuple(int(i) for i in bad[0])}: a flat or corrupt EEG channel"
        )
    return de.reshape(*de.shape[:-2], -1)


def band_powers(window: npt.NDArray[np.floating[Any]], recipe: TrainingRecipe = RECIPE) -> dict[str, float]:
    """A window's displayed band powers: each band's variance averaged over channels, by band name."""
    return band_powers_from(band_variance(window, recipe), recipe)


def band_powers_from(variance: npt.NDArray[np.float64], recipe: TrainingRecipe = RECIPE) -> dict[str, float]:
    """One window's displayed band powers from its ``(channels, bands)`` band variance."""
    mean_variance = variance.mean(axis=0)
    return {band.name: float(power) for band, power in zip(recipe.bands, mean_variance, strict=True)}


def windows_of(eeg: npt.NDArray[np.floating[Any]], recipe: TrainingRecipe = RECIPE) -> npt.NDArray[np.floating[Any]]:
    """Cut ``(trials, channels, samples)`` into non-overlapping windows: ``(trials, windows, channels, window_samples)``.

    Training and seeding both cut with this, so window ``w`` of trial ``e`` is the EEG of the
    stored Segment ``e * windows + w``.
    """
    n_trials, n_channels, n_samples = eeg.shape
    n_windows = n_samples // recipe.window_samples
    cut = eeg[..., : n_windows * recipe.window_samples].reshape(n_trials, n_channels, n_windows, -1)
    return cut.transpose(0, 2, 1, 3)
