"""What a judge is shown: a judge that sees the input gets the case's inputs plus its judged tools' returns.

Only the returns of the rubric's ``judged_tools``, and not any call's arguments: the arguments
are the model's own words, and shown as input they would let an invented figure pass as
grounded. A call that failed or was sent back returned nothing.
"""

import json
from dataclasses import replace

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from tests.evals.agent_run import AgentTurn, RecordedToolCall
from tests.evals.judges import Rubric, Verdict, judged_turn, verdict_of
from tests.factories import make_evaluator_context

_SEARCH = frozenset({"search_notes"})
_SEARCH_CALL = RecordedToolCall(
    name="search_notes", args={"query": "rent"}, returned={"results": [{"title": "Rent", "content": "$1,450 due 1st"}]}
)
_OTHER_CALLS = [
    RecordedToolCall(name="create_note", args={"title": "Tips", "content": "Tips bring in $150"}, returned={"id": 7}),
    RecordedToolCall(name="search_notes", args={"query": "x"}, returned="Search is down: $9,999.", outcome="failed"),
]


async def _prompt_the_judge_sees(rubric: Rubric, attributes: dict[str, object]) -> str:
    prompts: list[str] = []

    def judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        [request] = [message for message in messages if isinstance(message, ModelRequest)]
        prompts.extend(str(part.content) for part in request.parts if isinstance(part, UserPromptPart))
        verdict = json.dumps({"reason": "fine", "pass": True, "score": 1.0})
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, verdict)])

    context = make_evaluator_context(
        AgentTurn(prompt="When is rent due?"), "Rent is $1,450, due on the 1st.", attributes=attributes
    )
    await replace(rubric.judge(), model=FunctionModel(judge)).evaluate(context)
    [prompt] = prompts
    return prompt


async def test_a_judge_that_sees_the_input_is_shown_its_tools_returns_and_nothing_else_the_run_called() -> None:
    prompt = await _prompt_the_judge_sees(
        Rubric(name="grounded", text="Figures are grounded.", sees_input=True, judged_tools=_SEARCH),
        {"tool_calls": [*_OTHER_CALLS, _SEARCH_CALL]},
    )

    assert '"prompt":"When is rent due?"' in prompt
    assert (
        '"tool_returns":[{"name":"search_notes","returned":{"results":[{"title":"Rent","content":"$1,450 due 1st"}]}}]'
        in prompt
    )
    for unshown in ('"rent"', "Tips bring in", "9,999"):
        assert unshown not in prompt, unshown


async def test_a_run_with_no_judged_return_shows_an_empty_list() -> None:
    prompt = await _prompt_the_judge_sees(
        Rubric(name="grounded", text="Grounded.", sees_input=True, judged_tools=_SEARCH), {"tool_calls": _OTHER_CALLS}
    )

    assert '"tool_returns":[]' in prompt


async def test_a_judge_with_no_judged_tools_is_shown_the_inputs_alone() -> None:
    prompt = await _prompt_the_judge_sees(
        Rubric(name="grounded", text="Grounded.", sees_input=True), {"tool_calls": [_SEARCH_CALL]}
    )

    assert '"prompt":"When is rent due?"' in prompt and '"tool_returns":[]' in prompt


def test_judging_an_already_judged_turn_replaces_its_returns() -> None:
    turn = AgentTurn(prompt="When is rent due?")

    again = judged_turn(judged_turn(turn, [_SEARCH_CALL], _SEARCH), [], _SEARCH)

    assert (again.prompt, again.tool_returns) == ("When is rent due?", [])


async def test_a_judge_that_grades_the_output_alone_is_not_shown_the_tool_calls() -> None:
    prompt = await _prompt_the_judge_sees(Rubric(name="short", text="Short."), {"tool_calls": [_SEARCH_CALL]})

    assert "search_notes" not in prompt and "When is rent due?" not in prompt


async def test_a_verdict_is_read_off_the_same_evaluator_a_dataset_runs() -> None:
    prompts: list[str] = []

    def judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        [request] = [message for message in messages if isinstance(message, ModelRequest)]
        prompts.extend(str(part.content) for part in request.parts if isinstance(part, UserPromptPart))
        verdict = json.dumps({"reason": "quotes a figure the search did not return", "pass": False, "score": 0.0})
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, verdict)])

    rubric = Rubric(name="grounded", text="Grounded.", sees_input=True, judged_tools=_SEARCH)
    faked = replace(rubric.judge(), model=FunctionModel(judge))

    verdict = await verdict_of(
        faked, AgentTurn(prompt="When is rent due?"), "Rent is $1,450.", [*_OTHER_CALLS, _SEARCH_CALL]
    )

    assert verdict == Verdict(passed=False, reason="quotes a figure the search did not return")
    [prompt] = prompts
    assert '"tool_returns":[{"name":"search_notes"' in prompt and "Tips bring in" not in prompt
