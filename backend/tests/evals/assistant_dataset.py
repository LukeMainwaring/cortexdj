"""The assistant's eval cases: a prompt each, with the checks a good reply passes.

Each case names its evaluators: a rubric's ``judge()`` for what needs a reader, and a
deterministic evaluator for what does not. The greeting case is the copy-me pattern: add a
case per behavior the assistant must keep, and give every new rubric a good and a bad reply
in ``golden_set.py``.
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
