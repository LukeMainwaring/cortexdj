"""The golden set: hand-written replies that check the judges, not the agent.

For each judged behavior there is one reply a person would call good and one they would
call bad, written against the same inputs the agent cases use. Here the judge is the thing
under test: the task asks it for a verdict, ``expected_output`` is the human label, and
``EqualsExpected`` scores the match. No agent runs. If a judge misgrades one of these, its
scores on real replies mean little: sharpen the rubric or change ``EVAL_JUDGE_MODEL``.
"""

from pydantic import BaseModel
from pydantic_evals import Case, Dataset, set_eval_attribute
from pydantic_evals.evaluators import EqualsExpected

from tests.evals import judges
from tests.evals.agent_run import AgentOutput, AgentTurn, RecordedToolCall
from tests.evals.assistant_dataset import assistant_dataset

# Keyed by each rubric's own name, so a grader cannot be filed under another rubric's.
GRADERS: dict[str, judges.Grader] = {rubric.name: rubric.grade for rubric in [judges.FRIENDLY_GREETING]}


class GoldenReply(BaseModel):
    grader: str
    turn: AgentTurn
    reply: AgentOutput
    # What the run's tools returned before this reply, shown to judges that see the input.
    tool_calls: list[RecordedToolCall] = []


async def grade_golden_reply(golden: GoldenReply) -> bool:
    verdict = await GRADERS[golden.grader](golden.turn, golden.reply, golden.tool_calls)
    # Kept on the case so a misgrade can be read without re-running the judge.
    set_eval_attribute("judge_reason", verdict.reason)
    return verdict.passed


def _turn(dataset: Dataset[AgentTurn, AgentOutput, None], case_name: str) -> AgentTurn:
    [case] = [case for case in dataset.cases if case.name == case_name]
    return case.inputs


def _good_and_bad(
    grader: str,
    turn: AgentTurn,
    *,
    good: AgentOutput,
    bad: AgentOutput,
    tool_calls: list[RecordedToolCall] | None = None,
) -> list[Case[GoldenReply, bool, None]]:
    def golden(reply: AgentOutput) -> GoldenReply:
        return GoldenReply(grader=grader, turn=turn, reply=reply, tool_calls=tool_calls or [])

    return [
        Case(name=f"{grader}: good", inputs=golden(good), expected_output=True),
        Case(name=f"{grader}: bad", inputs=golden(bad), expected_output=False),
    ]


golden_set: Dataset[GoldenReply, bool, None] = Dataset(
    name="golden-set",
    cases=[
        *_good_and_bad(
            "friendly_greeting",
            _turn(assistant_dataset, "friendly_greeting"),
            good="Hello! It's lovely to meet you.",
            bad=(
                "Greetings. I am an assistant with access to your notes. I can list them, create new ones, "
                "and search them by meaning. State your request in full, including any constraints, and I "
                "will process it."
            ),
        ),
    ],
    evaluators=[EqualsExpected()],
)
