"""``train-classifier-modal``: train every participant's quadrant classifiers on Modal and write the checkpoint.

A second way to make the file ``train-classifier`` makes, not a replacement for it. Each
participant trains in its own Modal container, all at once, and the results come back to this
machine, which writes the same checkpoint to the same path. The only file that imports
``modal``: ``train_participant`` runs unchanged in the container, given the mounted Volume as
its data directory.

Run it on the host: ``modal`` is in the dev dependency group, which the backend container
does not install.

Usage (needs a Modal login and the DEAP Volume; ``DEVELOPMENT.md`` has the one-time setup):
    uv run --directory backend train-classifier-modal
    uv run --directory backend train-classifier-modal --participants 1 2 3
"""

import logging
from pathlib import Path

import modal

from cortexdj.ml.checkpoint import CHECKPOINT_PATH, ParticipantModels, QuadrantCheckpoint, save_checkpoint
from cortexdj.training.command import describe, log_participant_scores, parse_participants
from cortexdj.training.train import train_participant

# 4 parents: training/ → cortexdj/ → src/ → backend root, where pyproject.toml and uv.lock are.
# Found from this file, not the working directory, so the command runs from anywhere. Inside a
# container the path points nowhere, which is fine: the image is only built from this machine.
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DEAP_VOLUME_NAME = "cortexdj-deap"
_VOLUME_MOUNT = "/vol"
# The Volume holds the DEAP files under /deap; DEVELOPMENT.md has the upload command.
_DEAP_DIR_IN_CONTAINER = Path(_VOLUME_MOUNT) / "deap"

app = modal.App("cortexdj-train-classifier")

# The backend lockfile without the dev group, so Modal trains on the versions this machine
# does. uv_sync copies only pyproject.toml and uv.lock into the build, and the package itself is
# added at container start, so a code edit does not rebuild the dependency layer.
image = (
    modal.Image.debian_slim(python_version="3.14")
    .uv_sync(str(_BACKEND_ROOT), extra_options="--no-dev")
    .add_local_python_source("cortexdj")
)

# No create_if_missing: a run without the uploaded data should fail at start, not train on an
# empty Volume.
deap_volume = modal.Volume.from_name(DEAP_VOLUME_NAME)


@app.function(
    image=image,
    volumes={_VOLUME_MOUNT: deap_volume.read_only()},
    cpu=1,
    memory=2048,
    timeout=600,
    # For a preempted container. Modal retries every failure alike, so a mistake no retry can
    # fix (a Volume uploaded to the wrong path, a participant id with no file) also runs three
    # times before the run fails; at seconds a participant that is cheaper than telling them apart.
    retries=2,
)
def train_participant_on_modal(participant_id: int) -> ParticipantModels:
    """Train one participant in a Modal container and return their fold models to the caller.

    Returns ``ParticipantModels`` as it is: the caller is this repo on the same lockfile, so
    the class unpickles there. Do not write the checkpoint here; a container's disk is
    discarded, and one file holding every participant is assembled by ``main``.
    """
    return train_participant(participant_id, _DEAP_DIR_IN_CONTAINER)


def main() -> None:
    """Fan the participants out, then write the checkpoint only once every one has come back.

    A participant that still fails after its retries raises out of ``map``, so nothing is
    written and an existing checkpoint is left as it was. Kept in a function, never at module
    level: containers import this module, and a run started on import would start there too.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    participants = parse_participants("Train the quadrant classifier on Modal and write its checkpoint")

    # enable_output shows the image build and container logs, which app.run() hides by default.
    with modal.enable_output(), app.run():
        trained: dict[int, ParticipantModels] = dict(
            zip(participants, train_participant_on_modal.map(participants), strict=True)
        )

    for participant_id, models in trained.items():
        log_participant_scores(participant_id, models.metrics)
    checkpoint = QuadrantCheckpoint.of(trained)
    save_checkpoint(checkpoint)
    print(describe(checkpoint.summary, len(checkpoint.participants)))  # noqa: T201
    print(f"Wrote {CHECKPOINT_PATH}")  # noqa: T201
