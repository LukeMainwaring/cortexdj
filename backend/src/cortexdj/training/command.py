"""``train-classifier``: train every participant's quadrant classifiers and write the checkpoint.

Usage:
    uv run --directory backend train-classifier
    uv run --directory backend train-classifier --participants 1 2 3
"""

import argparse
import logging
from pathlib import Path

from cortexdj.core.paths import DEAP_DATA_DIR
from cortexdj.ml.checkpoint import CHECKPOINT_PATH, QuadrantCheckpoint, ScoreSummary, save_checkpoint
from cortexdj.ml.recipe import RECIPE, TrainingRecipe
from cortexdj.training.train import train_participant

logger = logging.getLogger(__name__)


def train_classifier(
    participants: list[int],
    data_dir: Path = DEAP_DATA_DIR,
    checkpoint_path: Path = CHECKPOINT_PATH,
    recipe: TrainingRecipe = RECIPE,
) -> QuadrantCheckpoint:
    """Train ``participants`` and write a checkpoint holding only them, overwriting any file at ``checkpoint_path``.

    A subset run does not merge into an existing checkpoint; seeding a participant the file
    lacks is an error that says to run this again.
    """
    trained = {}
    for participant_id in participants:
        trained[participant_id] = train_participant(participant_id, data_dir, recipe)
        metrics = trained[participant_id].metrics
        logger.info(
            "Participant %d: arousal %.3f, valence %.3f, mean %.3f",
            participant_id,
            metrics.arousal,
            metrics.valence,
            metrics.mean,
        )
    checkpoint = QuadrantCheckpoint.of(trained, recipe)
    save_checkpoint(checkpoint, checkpoint_path)
    return checkpoint


def describe(summary: ScoreSummary, n_participants: int) -> str:
    """The run's result in three lines: balanced accuracy per axis and their mean, ± the spread across participants."""
    return "\n".join(
        [
            f"Balanced accuracy over {n_participants} participants (chance 0.50), mean ± std across participants:",
            f"  arousal  {summary.arousal.mean:.3f} ± {summary.arousal.std:.3f}",
            f"  valence  {summary.valence.mean:.3f} ± {summary.valence.std:.3f}",
            f"  mean     {summary.mean.mean:.3f} ± {summary.mean.std:.3f}",
        ]
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train the quadrant classifier and write its checkpoint")
    parser.add_argument("--participants", nargs="+", type=int, default=list(range(1, 33)))
    args = parser.parse_args()

    checkpoint = train_classifier(args.participants)
    print(describe(checkpoint.summary, len(checkpoint.participants)))  # noqa: T201
    print(f"Wrote {CHECKPOINT_PATH}")  # noqa: T201
