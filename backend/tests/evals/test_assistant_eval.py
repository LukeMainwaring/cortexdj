"""Real-model eval of the assistant as it ships. A measurement, not a gate.

Excluded from the default run and from PR CI: it needs a real provider key and costs money.
Run it with ``./backend/scripts/run-evals.sh``, or on a schedule (.github/workflows/eval.yml).
The test prints its report and fails only when a case could not be run or scored at all.
Pass rates are for reading and for the baseline in ``docs/evals/``, not for asserting on: a
threshold needs a known spread first.

Environment variables: ``EVAL_REPEAT`` is how many times each case runs (default 3, since one
run of a model says little; a before-and-after comparison of a prompt change wants 5 or more,
see DEVELOPMENT.md); ``EVAL_CASES`` is a comma-separated list of case names to run instead of
the whole dataset, for a cheap run while changing a prompt.

Each run carries the agent's model, provider, and reasoning effort, the judge's model, and the
commit it tested as experiment metadata, so runs compare across model swaps in Logfire's
Experiments view (ADR 0003) and a run graded by a different judge is not mistaken for a like one.

Import-safe: the dataset is pure data (no model call) and the agent is built inside the task,
so collecting this module needs no key.
"""

import subprocess

from pydantic_ai.models import Model

from cortexdj.agents.registry import ASSISTANT, get_agent
from cortexdj.core.config import get_settings
from tests.evals.agent_run import describe_tool_call, run_assistant
from tests.evals.assistant_dataset import assistant_dataset
from tests.evals.judges import judge_model
from tests.evals.run_options import TASK_TIMEOUT_SECONDS, eval_repeat, selected_cases

REPORT_WIDTH = 200


def _experiment_metadata() -> dict[str, str]:
    """What the run measured with: the assistant's model and effort, read off the built agent, the judge, and the commit."""
    model = get_agent(ASSISTANT).model
    assert isinstance(model, Model), f"{ASSISTANT} was built without a concrete model"
    return {
        "model": model.model_name,
        "provider": model.system,
        "reasoning_effort": get_settings().AGENT_REASONING_EFFORT or "provider default",
        "judge_model": judge_model().model_name,
        "git_sha": _git_sha(),
    }


def _git_sha() -> str:
    """HEAD's short sha. Uncommitted edits are not marked: the experiment's timestamp tells those runs apart."""
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
    except OSError, subprocess.CalledProcessError:
        return "unknown"
    return result.stdout.strip()


async def test_assistant_eval_runs_every_case() -> None:
    selected = selected_cases(assistant_dataset, lambda case: [case.name])

    report = await selected.evaluate(
        run_assistant,
        repeat=eval_repeat(),
        max_concurrency=4,
        task_timeout=TASK_TIMEOUT_SECONDS,
        progress=False,
        metadata=_experiment_metadata(),
    )
    report.print(width=REPORT_WIDTH, include_input=True, include_output=True, include_reasons=True)
    # The report table has no column for attributes, so the tools each run called print below it.
    for case in report.cases:
        for call in case.attributes.get("tool_calls", []):
            print(f"{case.name} {describe_tool_call(call)}")

    assert not report.failures, [failure.error_message for failure in report.failures]
    unscored = [failure for case in report.cases for failure in case.evaluator_failures]
    assert not unscored, [f"{failure.name}: {failure.error_message}" for failure in unscored]
