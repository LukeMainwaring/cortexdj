"""The run's tool calls as the eval task records them: paired with what each tool answered."""

from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from tests.evals.agent_run import RecordedToolCall, describe_tool_call, record_tool_calls


class _Answer(BaseModel):
    total: int


def test_each_call_is_paired_with_its_return_in_call_order() -> None:
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart("Which tracks relax me?")]),
        ModelResponse(
            parts=[
                ToolCallPart("find_relaxing_tracks", {"limit": 5}, tool_call_id="s1"),
                ToolCallPart("set_brain_context", '{"dominant_mood": "relaxed", "avg_arousal": 2}', tool_call_id="c1"),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart("find_relaxing_tracks", {"tracks": []}, tool_call_id="s1"),
                ToolReturnPart(
                    "set_brain_context", "avg_arousal is out of range", tool_call_id="c1", outcome="retried"
                ),
            ]
        ),
        ModelResponse(parts=[ToolCallPart("list_sessions", {}, tool_call_id="l1")]),
        ModelRequest(parts=[ToolReturnPart("list_sessions", _Answer(total=2), tool_call_id="l1")]),
        ModelResponse(parts=[ToolCallPart("set_brain_context", {"dominant_mood": "calm"}, tool_call_id="c2")]),
        ModelRequest(
            parts=[ToolReturnPart("set_brain_context", "The database is down.", tool_call_id="c2", outcome="failed")]
        ),
        ModelResponse(parts=[TextPart("No relaxing tracks yet.")]),
    ]

    assert record_tool_calls(messages) == [
        RecordedToolCall(name="find_relaxing_tracks", args={"limit": 5}, returned={"tracks": []}),
        RecordedToolCall(
            name="set_brain_context",
            args={"dominant_mood": "relaxed", "avg_arousal": 2},
            returned="avg_arousal is out of range",
            outcome="retried",
        ),
        RecordedToolCall(name="list_sessions", args={}, returned={"total": 2}),
        RecordedToolCall(
            name="set_brain_context", args={"dominant_mood": "calm"}, returned="The database is down.", outcome="failed"
        ),
    ]


def test_an_output_tool_is_not_recorded_because_its_arguments_are_the_output() -> None:
    messages: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart("final_result", {"summary": "x"}, tool_call_id="o1")]),
        ModelRequest(parts=[ToolReturnPart("final_result", "Final result processed.", tool_call_id="o1")]),
    ]

    assert record_tool_calls(messages, output_tools=("final_result",)) == []
    assert [call.name for call in record_tool_calls(messages)] == ["final_result"]


def test_a_call_that_never_got_an_answer_is_recorded_without_one() -> None:
    messages: list[ModelMessage] = [ModelResponse(parts=[ToolCallPart("find_relaxing_tracks", {}, tool_call_id="s1")])]

    assert record_tool_calls(messages) == [
        RecordedToolCall(name="find_relaxing_tracks", args={}, returned=None, outcome="no answer")
    ]


def test_a_recorded_call_prints_as_one_line_with_a_long_return_cut_short() -> None:
    call = RecordedToolCall(name="find_relaxing_tracks", args={"limit": 5}, returned={"body": "x" * 5000})

    line = describe_tool_call(call)

    assert line.startswith('find_relaxing_tracks({"limit":5}) -> {"body":"xxx')
    assert line.endswith("... (5011 characters)")
    assert len(line) < 2100
    failed = RecordedToolCall(name="set_brain_context", args={}, returned="down", outcome="failed")
    assert describe_tool_call(failed) == 'set_brain_context({}) failed -> "down"'
