"""EEG windowing for the contrastive EEG↔CLAP encoder.

Turns a DEAP trial into the 4-second, 200Hz windows the encoder's CBraMod
backbone takes as input.
"""

import numpy as np
import numpy.typing as npt
from scipy.signal import resample

from cortexdj.ml.dataset import CBRAMOD_SAMPLING_RATE, CBRAMOD_SCALE_FACTOR, SEGMENT_SAMPLES

CBRAMOD_SEGMENT_SAMPLES = CBRAMOD_SAMPLING_RATE * 4  # 800 samples at 200Hz


def trial_to_eeg_windows(
    trial_data: npt.NDArray[np.float32],
    *,
    source_segment_samples: int = SEGMENT_SAMPLES,
    target_segment_samples: int = CBRAMOD_SEGMENT_SAMPLES,
) -> npt.NDArray[np.float32]:
    """Slice a DEAP trial into non-overlapping 4s windows resampled for CBraMod.

    The contrastive encoder was trained on windows cut this way, so retrieval
    must cut them the same way: do not change the window length, the resampling
    or the scaling without retraining. `trial_data` is expected as (n_channels, n_samples) at DEAP's 128Hz sampling rate;
    returns (n_windows, n_channels, target_segment_samples) at 200Hz, scaled by
    `CBRAMOD_SCALE_FACTOR` to match the pretrained input distribution.
    """
    n_samples = trial_data.shape[1]
    n_segments = n_samples // source_segment_samples
    windows: list[npt.NDArray[np.float32]] = []
    for seg_idx in range(n_segments):
        start = seg_idx * source_segment_samples
        end = start + source_segment_samples
        segment = trial_data[:, start:end]  # (n_channels, source_segment_samples)
        resampled = resample(segment, target_segment_samples, axis=1).astype(np.float32)
        windows.append(resampled * CBRAMOD_SCALE_FACTOR)
    if not windows:
        return np.zeros((0, trial_data.shape[0], target_segment_samples), dtype=np.float32)
    return np.stack(windows, axis=0)
