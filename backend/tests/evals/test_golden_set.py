"""The golden set is a real test: every hand-labeled reply must be graded the way it is labeled.

Unlike the agent eval, this one asserts on the result. It needs the judge model's provider key
and no database. ``EVAL_CASES``, as in ``test_assistant_eval.py``, is a comma-separated list
that narrows the run; here each entry is a case name ("friendly_greeting: good") or a grader
name ("friendly_greeting"), for a cheap run after changing one rubric.
"""

from tests.evals.golden_set import golden_set, grade_golden_reply
from tests.evals.run_options import TASK_TIMEOUT_SECONDS, selected_cases

REPORT_WIDTH = 160


async def test_judges_grade_the_golden_set_the_way_a_person_would() -> None:
    selected = selected_cases(golden_set, lambda case: [case.name, case.inputs.grader])

    # Once, not `EVAL_REPEAT` times: the labels are fixed, so a repeat only measures judge noise.
    report = await selected.evaluate(
        grade_golden_reply, max_concurrency=5, task_timeout=TASK_TIMEOUT_SECONDS, progress=False
    )
    report.print(width=REPORT_WIDTH, include_output=True, include_expected_output=True)

    assert not report.failures, [failure.error_message for failure in report.failures]
    misgraded = [
        f"{case.name}: judge said {case.output}, because: {case.attributes['judge_reason']}"
        for case in report.cases
        if case.output != case.expected_output
    ]
    assert not misgraded, "\n".join(misgraded)
