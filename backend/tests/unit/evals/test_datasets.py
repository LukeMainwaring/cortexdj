"""The eval datasets hold what the baseline reports on, and every judge in them is calibrated.

Importing the datasets here is itself a check: a dataset that ran a model, or an eval module
that needed the database at import, would fail in the unit tier. The rest enforces what ADR
0004 states: a judge runs on the pinned judge model, and a rubric with no golden pair is not
calibrated.
"""

from pydantic_evals.evaluators import LLMJudge

from tests.evals.assistant_dataset import assistant_dataset
from tests.evals.golden_set import GRADERS, golden_set
from tests.evals.judges import judge_model


def _model_judges() -> list[LLMJudge]:
    return [
        evaluator
        for evaluators in (assistant_dataset.evaluators, *(case.evaluators for case in assistant_dataset.cases))
        for evaluator in evaluators
        if isinstance(evaluator, LLMJudge)
    ]


def _rubric_name(judge: LLMJudge) -> str | None:
    """The name ``Rubric.judge()`` gives a judge's verdict; None for a judge built any other way."""
    if judge.assertion is False:
        return None
    return judge.assertion.get("evaluation_name")


def test_the_assistant_dataset_holds_the_baseline_cases() -> None:
    assert [case.name for case in assistant_dataset.cases] == ["friendly_greeting"]


def test_every_judge_runs_on_the_pinned_judge_model() -> None:
    judges = _model_judges()
    assert judges, "the dataset is the copy-me pattern for a judged behavior; it should carry one"
    assert all(judge.model is judge_model() for judge in judges)


def test_every_judged_behavior_has_a_good_and_a_bad_golden_reply() -> None:
    judged = [_rubric_name(judge) for judge in _model_judges()]
    assert None not in judged, "build judges with Rubric.judge(): a judge with no named verdict has no golden pair"
    assert set(judged) == set(GRADERS), "a rubric with no golden pair is not calibrated (ADR 0004)"

    labels: dict[str, list[bool | None]] = {}
    for case in golden_set.cases:
        labels.setdefault(case.inputs.grader, []).append(case.expected_output)
    assert labels == {grader: [True, False] for grader in GRADERS}
