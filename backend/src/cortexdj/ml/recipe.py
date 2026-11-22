"""The training recipe: every setting the quadrant classifier is trained and read with."""

from dataclasses import dataclass

from cortexdj.ml.dataset import HIGH_RATING


@dataclass(frozen=True)
class Band:
    """One frequency band, ``low`` to ``high`` Hz."""

    name: str
    low: float
    high: float


@dataclass(frozen=True)
class TrainingRecipe:
    """The constants one quadrant-classifier checkpoint is made with, shared by every participant.

    The settings are fixed and untuned: they were chosen once, not searched on the folds the
    headline number is scored on, because comparing many recipes on the same folds drifts that
    number upward. They are part of the model, not per-machine configuration, so do not move
    them into settings or env vars. A checkpoint stores the recipe it was made with, and loading
    refuses one whose recipe differs from this value.

    ``weight_decay`` is strong on purpose. Full-batch AdamW holds each weight near
    ``1 / weight_decay`` at most, so 3.0 keeps a weight on a standardized feature to about a
    third, and few held-out scores fall below 0.1 or above 0.9; a weak decay lets the scores pile
    up at 0 and 1 without scoring any better.
    """

    bands: tuple[Band, ...] = (
        Band("theta", 4.0, 8.0),
        Band("alpha", 8.0, 13.0),
        Band("beta", 13.0, 30.0),
        Band("gamma", 30.0, 45.0),
    )
    filter_order: int = 4
    sample_rate: int = 128
    window_seconds: int = 4
    label_threshold: float = HIGH_RATING
    n_folds: int = 5
    seed: int = 0
    learning_rate: float = 0.01
    epochs: int = 300
    weight_decay: float = 3.0

    @property
    def window_samples(self) -> int:
        """Samples in one window: the same slice of EEG as one stored Segment."""
        return self.sample_rate * self.window_seconds


RECIPE = TrainingRecipe()
