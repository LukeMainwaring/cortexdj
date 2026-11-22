"""The band features the quadrant classifier reads, and the band powers a Segment shows.

Both come from one band-pass of the 4-second window, so the model reads the log of what the
chart draws. Training and seeding both call these functions; do not compute either a second way.
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
    """The classifier's input: half the log of each band variance, ``(..., 32, 512)`` to ``(..., 128)``.

    Channel-major: feature ``4 * c + b`` is channel ``c`` in band ``b``. The Gaussian constant
    ``0.5 * log(2 pi e)`` is left out because the model standardizes its input.
    """
    variance = band_variance(windows, recipe)
    de: npt.NDArray[np.float64] = 0.5 * np.log(variance)
    return de.reshape(*de.shape[:-2], -1)


def band_powers(window: npt.NDArray[np.floating[Any]], recipe: TrainingRecipe = RECIPE) -> dict[str, float]:
    """A window's displayed band powers: each band's variance averaged over channels, by band name."""
    mean_variance = band_variance(window, recipe).mean(axis=0)
    return {band.name: float(power) for band, power in zip(recipe.bands, mean_variance, strict=True)}
