"""The assistant's eval cases: a prompt each, with the checks a good reply passes.

Each case names its evaluators. The greeting case is the copy-me pattern: a rubric's
``judge()`` for a behavior that needs a reader. Add a case per behavior the assistant must
keep, and give every new rubric a good and a bad reply in ``golden_set.py``. A behavior that
needs no reader takes a deterministic evaluator from ``evaluators.py``; no case uses one yet.
"""

from pydantic_evals import Case, Dataset

from tests.evals.agent_run import AgentOutput, AgentTurn
from tests.evals.judges import FRIENDLY_GREETING

assistant_dataset: Dataset[AgentTurn, AgentOutput, None] = Dataset(
    name="assistant",
    cases=[
        Case(
            name="friendly_greeting",
            inputs=AgentTurn(prompt="Say hello in one short, friendly sentence."),
            evaluators=(FRIENDLY_GREETING.judge(),),
        ),
    ],
)
