"""Generate synthetic EEG data for development and testing.

Produces multi-channel EEG-like signals with controlled frequency band
power ratios, yielding known arousal/valence labels for training.

Output format mirrors DEAP structure:
  - data: (n_trials, n_channels, n_samples)
  - labels: (n_trials, 4) — [valence, arousal, dominance, liking]

Usage:
    uv run generate-synthetic
    uv run generate-synthetic --participants 16 --trials 20
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from cortexdj.core.paths import SYNTHETIC_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NUM_CHANNELS = 32
SAMPLING_RATE = 128
TRIAL_DURATION_SECONDS = 60  # 60 seconds per trial
TRIAL_SAMPLES = SAMPLING_RATE * TRIAL_DURATION_SECONDS


def _generate_eeg_signal(
    n_channels: int,
    n_samples: int,
    fs: int,
    *,
    alpha_power: float = 1.0,
    beta_power: float = 1.0,
    theta_power: float = 1.0,
    delta_power: float = 0.5,
    gamma_power: float = 0.3,
    noise_level: float = 0.5,
) -> npt.NDArray[np.floating[Any]]:
    """Generate synthetic multi-channel EEG with controlled band powers.

    Each frequency band is a sum of sinusoids at frequencies within the band,
    scaled by the specified power. Gaussian noise is added for realism.
    """
    t = np.arange(n_samples) / fs
    signal = np.zeros((n_channels, n_samples))

    band_configs = [
        (1.0, 4.0, delta_power),
        (4.0, 8.0, theta_power),
        (8.0, 14.0, alpha_power),
        (14.0, 30.0, beta_power),
        (30.0, 40.0, gamma_power),
    ]

    rng = np.random.default_rng()

    for ch in range(n_channels):
        ch_signal = np.zeros(n_samples)
        for low, high, power in band_configs:
            for _ in range(3):
                freq = rng.uniform(low, high)
                phase = rng.uniform(0, 2 * np.pi)
                amplitude = power * rng.uniform(0.5, 1.5)
                ch_signal += amplitude * np.sin(2 * np.pi * freq * t + phase)

        ch_signal += noise_level * rng.normal(size=n_samples)
        signal[ch] = ch_signal

    return signal


def _generate_participant(
    participant_id: int,
    n_trials: int,
    rng: np.random.Generator,
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
]:
    """Generate synthetic EEG data for one participant.

    Band power profiles determine the arousal/valence labels:
      - High alpha -> relaxed (low arousal, high valence)
      - High beta -> stressed (high arousal, low valence)
      - High alpha + theta -> calm (low arousal, low valence)
      - High beta + alpha -> excited (high arousal, high valence)
    """
    all_data = np.zeros((n_trials, NUM_CHANNELS, TRIAL_SAMPLES))
    all_labels = np.zeros((n_trials, 4))

    for trial in range(n_trials):
        profile = rng.choice(["relaxed", "stressed", "calm", "excited"])

        if profile == "relaxed":
            alpha, beta, theta = 3.0, 0.5, 1.0
            valence, arousal = rng.uniform(6, 9), rng.uniform(1, 4)
        elif profile == "stressed":
            alpha, beta, theta = 0.5, 3.0, 0.5
            valence, arousal = rng.uniform(1, 4), rng.uniform(6, 9)
        elif profile == "calm":
            alpha, beta, theta = 1.5, 0.5, 2.5
            valence, arousal = rng.uniform(2, 4.5), rng.uniform(1, 4)
        else:  # excited
            alpha, beta, theta = 2.0, 2.5, 0.5
            valence, arousal = rng.uniform(6, 9), rng.uniform(6, 9)

        signal = _generate_eeg_signal(
            NUM_CHANNELS,
            TRIAL_SAMPLES,
            SAMPLING_RATE,
            alpha_power=alpha,
            beta_power=beta,
            theta_power=theta,
        )

        all_data[trial] = signal
        all_labels[trial] = [valence, arousal, rng.uniform(3, 7), rng.uniform(3, 7)]

    return all_data, all_labels


def generate(*, n_participants: int = 32, n_trials: int = 40) -> None:
    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    logger.info(f"Generating synthetic EEG data: {n_participants} participants, {n_trials} trials each")

    for p in range(1, n_participants + 1):
        data, labels = _generate_participant(p, n_trials, rng)
        output_path = SYNTHETIC_DATA_DIR / f"s{p:02d}.npz"
        np.savez_compressed(output_path, data=data, labels=labels)
        logger.info(f"  [{p}/{n_participants}] Saved {output_path.name} — {data.shape}")

    logger.info(f"Done! Generated {n_participants} files in {SYNTHETIC_DATA_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic EEG data")
    parser.add_argument("--participants", type=int, default=32, help="Number of participants")
    parser.add_argument("--trials", type=int, default=40, help="Trials per participant")
    args = parser.parse_args()

    generate(n_participants=args.participants, n_trials=args.trials)
