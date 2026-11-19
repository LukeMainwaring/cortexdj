"""The environment variables that shape an eval run, read once and checked.

A value that matches nothing must not pass as green: a zero ``EVAL_REPEAT`` fails, and an
``EVAL_CASES`` that names nothing in a dataset skips that dataset's test with the names in the
message. A skip, not a failure, because one ``EVAL_CASES`` reaches every dataset in the run, and a
name that exists in one of them matches nothing in the others.
"""

import os
from collections.abc import Callable, Collection

import pytest
from pydantic_evals import Case, Dataset

# Per call of a dataset's task, so one hung case fails on its own instead of holding the run
# until the job's timeout. Evaluators run after the task, outside it: the assistant eval's
# judges are not bounded by this, the golden set's judge (called inside its task) is.
TASK_TIMEOUT_SECONDS = 120


def eval_repeat(default: int = 3) -> int:
    """``EVAL_REPEAT``: how many times each case runs; blank or unset keeps ``default``."""
    raw = os.environ.get("EVAL_REPEAT", "").strip()
    if not raw:
        return default
    if not raw.isdigit() or int(raw) < 1:
        pytest.fail(f"EVAL_REPEAT must be a whole number of 1 or more, not {raw!r}")
    return int(raw)


def selected_cases[InputsT, OutputT, MetadataT](
    dataset: Dataset[InputsT, OutputT, MetadataT],
    names_of: Callable[[Case[InputsT, OutputT, MetadataT]], Collection[str | None]],
) -> Dataset[InputsT, OutputT, MetadataT]:
    """The dataset narrowed to the cases ``EVAL_CASES`` names, or the whole dataset when it is unset.

    ``names_of`` gives the names a case answers to (its own name, and for the golden set its
    grader's too; an unnamed case answers to nothing). When no case answers, the test skips.
    """
    names = {name.strip() for name in os.environ.get("EVAL_CASES", "").split(",") if name.strip()}
    if not names:
        return dataset
    cases = [case for case in dataset.cases if names & set(names_of(case))]
    if not cases:
        pytest.skip(f"EVAL_CASES names nothing in {dataset.name}: {sorted(names)}")
    return dataset.model_copy(update={"cases": cases})
