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

_TRACKS = frozenset({"find_relaxing_tracks"})
_TRACKS_CALL = RecordedToolCall(
    name="find_relaxing_tracks",
    args={"limit": 3},
    returned={"tracks": [{"title": "Clair de Lune", "avg_valence": 0.71}]},
)
_OTHER_CALLS = [
    RecordedToolCall(
        name="set_brain_context", args={"dominant_mood": "calm", "avg_valence": 0.55}, returned="Brain context set."
    ),
    RecordedToolCall(
        name="find_relaxing_tracks", args={"limit": 9}, returned="Lookup is down: 0.99.", outcome="failed"
    ),
]


async def _prompt_the_judge_sees(rubric: Rubric, attributes: dict[str, object]) -> str:
    prompts: list[str] = []

    def judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        [request] = [message for message in messages if isinstance(message, ModelRequest)]
        prompts.extend(str(part.content) for part in request.parts if isinstance(part, UserPromptPart))
        verdict = json.dumps({"reason": "fine", "pass": True, "score": 1.0})
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, verdict)])

    context = make_evaluator_context(
        AgentTurn(prompt="Which track relaxed me?"), "Clair de Lune, at a valence of 0.71.", attributes=attributes
    )
    await replace(rubric.judge(), model=FunctionModel(judge)).evaluate(context)
    [prompt] = prompts
    return prompt


async def test_a_judge_that_sees_the_input_is_shown_its_tools_returns_and_nothing_else_the_run_called() -> None:
    prompt = await _prompt_the_judge_sees(
        Rubric(name="grounded", text="Figures are grounded.", sees_input=True, judged_tools=_TRACKS),
        {"tool_calls": [*_OTHER_CALLS, _TRACKS_CALL]},
    )

    assert '"prompt":"Which track relaxed me?"' in prompt
    assert (
        '"tool_returns":[{"name":"find_relaxing_tracks","returned":{"tracks":[{"title":"Clair de Lune","avg_valence":0.71}]}}]'
        in prompt
    )
    for unshown in ('"limit"', "0.55", "0.99"):
        assert unshown not in prompt, unshown


async def test_a_run_with_no_judged_return_shows_an_empty_list() -> None:
    prompt = await _prompt_the_judge_sees(
        Rubric(name="grounded", text="Grounded.", sees_input=True, judged_tools=_TRACKS), {"tool_calls": _OTHER_CALLS}
    )

    assert '"tool_returns":[]' in prompt


async def test_a_judge_with_no_judged_tools_is_shown_the_inputs_alone() -> None:
    prompt = await _prompt_the_judge_sees(
        Rubric(name="grounded", text="Grounded.", sees_input=True), {"tool_calls": [_TRACKS_CALL]}
    )

    assert '"prompt":"Which track relaxed me?"' in prompt and '"tool_returns":[]' in prompt


def test_judging_an_already_judged_turn_replaces_its_returns() -> None:
    turn = AgentTurn(prompt="Which track relaxed me?")

    again = judged_turn(judged_turn(turn, [_TRACKS_CALL], _TRACKS), [], _TRACKS)

    assert (again.prompt, again.tool_returns) == ("Which track relaxed me?", [])


async def test_a_judge_that_grades_the_output_alone_is_not_shown_the_tool_calls() -> None:
    prompt = await _prompt_the_judge_sees(Rubric(name="short", text="Short."), {"tool_calls": [_TRACKS_CALL]})

    assert "find_relaxing_tracks" not in prompt and "Which track relaxed me?" not in prompt


async def test_a_verdict_is_read_off_the_same_evaluator_a_dataset_runs() -> None:
    prompts: list[str] = []

    def judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        [request] = [message for message in messages if isinstance(message, ModelRequest)]
        prompts.extend(str(part.content) for part in request.parts if isinstance(part, UserPromptPart))
        verdict = json.dumps({"reason": "quotes a figure the tool did not return", "pass": False, "score": 0.0})
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, verdict)])

    rubric = Rubric(name="grounded", text="Grounded.", sees_input=True, judged_tools=_TRACKS)
    faked = replace(rubric.judge(), model=FunctionModel(judge))

    verdict = await verdict_of(
        faked, AgentTurn(prompt="Which track relaxed me?"), "Clair de Lune, at 0.71.", [*_OTHER_CALLS, _TRACKS_CALL]
    )

    assert verdict == Verdict(passed=False, reason="quotes a figure the tool did not return")
    [prompt] = prompts
    assert '"tool_returns":[{"name":"find_relaxing_tracks"' in prompt and "0.55" not in prompt
