"""The model-graded checks: one binary rubric per behavior.

Every judge runs on ``EVAL_JUDGE_MODEL``, which is pinned and does not follow ``AGENT_MODEL``
(``core/config.py`` and ADR 0004 say why). The model is built once (``judge_model``) and passed
to each judge rather than set with ``set_default_judge_model``, which changes a library global
at import.

A rubric is a statement that is true of a good reply; the judge passes the reply when the
statement holds, and gives a reason either way. Each rubric names one behavior, so a failing
row in the report says which behavior broke. A rubric is used two ways through one evaluator:
``judge()`` is the evaluator a dataset attaches to its cases, and ``grade()`` runs that same
evaluator over one reply for a verdict, which is how the golden set (``golden_set.py``) checks
the judge against hand-labeled replies. Change a rubric's wording and the golden set tells you
whether the judge still grades the way a person would.

A rubric that sees the input is shown a ``JudgedTurn``: the case's inputs plus what the run
got back from the rubric's ``judged_tools``, so "in the input" covers the figures the reply
quotes from a tool. ``judge()`` builds it from the run's ``tool_calls`` attribute; the golden
set builds it from the calls written into each golden reply.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from functools import cache

from pydantic import BaseModel
from pydantic_ai.models import Model, infer_model
from pydantic_evals.evaluators import EvaluationReason, EvaluatorContext, EvaluatorOutput, LLMJudge, OutputConfig

from cortexdj.core.config import get_settings
from tests.evals.agent_run import AgentOutput, AgentTurn, JudgedToolReturn, JudgedTurn, RecordedToolCall
from tests.factories import make_evaluator_context


@cache
def judge_model() -> Model:
    """The judge's model, built once per process: given a name instead, every judge call builds its own provider client."""
    return infer_model(get_settings().EVAL_JUDGE_MODEL)


class Verdict(BaseModel):
    passed: bool
    reason: str


# Tools whose returns a rubric that sees the input is shown by default. Empty here: the shipped
# rubric grades the output alone. Name the tools whose returns hold the facts a reply quotes (a
# calculation, a query over stored rows), and leave out any tool whose return echoes the model's
# own arguments: a figure the model invented would then be "in the input".
JUDGED_TOOLS: frozenset[str] = frozenset()


def judged_turn(
    turn: AgentTurn, tool_calls: list[RecordedToolCall], tools: frozenset[str] = JUDGED_TOOLS
) -> JudgedTurn:
    """The case's inputs with what ``tools`` returned beside them; a call that failed returned nothing."""
    returns = [
        JudgedToolReturn(name=call.name, returned=call.returned)
        for call in tool_calls
        if call.name in tools and call.outcome == "success"
    ]
    return JudgedTurn.model_validate({**turn.model_dump(), "tool_returns": returns})


# A dataclass, not a Pydantic model, because LLMJudge is one and `dataclasses.replace` builds the subclass.
@dataclass(repr=False)
class _TurnAndToolsJudge(LLMJudge):
    """An ``LLMJudge`` whose input is the turn plus what its rubric's tools returned, not the turn alone."""

    judged_tools: frozenset[str] = JUDGED_TOOLS

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
        assert isinstance(ctx.inputs, AgentTurn)
        turn = judged_turn(ctx.inputs, ctx.attributes.get("tool_calls", []), self.judged_tools)
        return await super().evaluate(replace(ctx, inputs=turn))


class Rubric(BaseModel, frozen=True):
    name: str
    text: str
    # Shown the case's inputs and what the run's judged tools returned as well as the output.
    # Only for rubrics that cannot be graded from the output alone.
    sees_input: bool = False
    # Whose returns a rubric that sees the input is shown; ignored when it does not.
    judged_tools: frozenset[str] = JUDGED_TOOLS

    def judge(self) -> LLMJudge:
        assertion = OutputConfig(evaluation_name=self.name, include_reason=True)
        if not self.sees_input:
            return LLMJudge(rubric=self.text, model=judge_model(), assertion=assertion)
        return _TurnAndToolsJudge(
            rubric=self.text,
            model=judge_model(),
            include_input=True,
            assertion=assertion,
            judged_tools=self.judged_tools,
        )

    async def grade(self, turn: AgentTurn, output: AgentOutput, tool_calls: list[RecordedToolCall]) -> Verdict:
        """Grade one reply with this rubric's evaluator; ``tool_calls`` may hold any tool's, and only its judged tools are shown."""
        return await verdict_of(self.judge(), turn, output, tool_calls)


async def verdict_of(
    judge: LLMJudge, turn: AgentTurn, output: AgentOutput, tool_calls: list[RecordedToolCall]
) -> Verdict:
    """Run ``judge`` over one reply as a dataset would, with no agent run behind it (so no span tree)."""
    ctx = make_evaluator_context(turn, output, attributes={"tool_calls": tool_calls})
    graded = await judge.evaluate(ctx)
    assert isinstance(graded, dict), "the judge is built with a named assertion, which reports as a dict"
    [result] = graded.values()
    assert isinstance(result, EvaluationReason), "the judge is built with include_reason=True"
    return Verdict(passed=bool(result.value), reason=result.reason or "")


FRIENDLY_GREETING = Rubric(
    name="friendly_greeting",
    text=(
        "The reply is a brief, friendly greeting: one or two short sentences that greet the user warmly. "
        "Fail if it runs longer than that, lists what the assistant can do, asks for instructions, or "
        "reads as curt or formal rather than friendly."
    ),
)


Grader = Callable[[AgentTurn, AgentOutput, list[RecordedToolCall]], Awaitable[Verdict]]
