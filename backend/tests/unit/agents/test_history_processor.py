"""History processor: compact large prior-turn tool results once the window fills, keep the current turn."""

from typing import Any, cast

from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from cortexdj.agents.history_processor import (
    COMPACT_ABOVE_WINDOW_FRACTION,
    build_tool_result_summarizer,
    compact_prior_tool_results,
)

SUMMARIZABLE = frozenset({"list_notes"})


def _notes_payload(n: int) -> dict[str, object]:
    return {"notes": [{"id": str(i), "title": f"n{i}", "content": "x" * 100} for i in range(n)], "total": n}


def test_large_prior_list_result_is_summarized() -> None:
    big = _notes_payload(50)  # well over LARGE_RESULT_THRESHOLD once serialized
    prior = ModelRequest(parts=[ToolReturnPart(tool_name="list_notes", content=big, tool_call_id="c1")])
    last = ModelRequest(parts=[ToolReturnPart(tool_name="list_notes", content=big, tool_call_id="c2")])

    out = compact_prior_tool_results([prior, last], SUMMARIZABLE)

    assert len(out) == 2  # the documented invariant: parts are rewritten, messages are not dropped
    summarized = out[0]
    assert isinstance(summarized, ModelRequest)
    part = summarized.parts[0]
    assert isinstance(part, ToolReturnPart)
    content = cast(dict[str, Any], part.content)
    assert content["_summarized"] is True
    assert content["tool"] == "list_notes"
    assert len(content["notes"]) == 5  # sample of the list
    assert content["total"] == 50  # scalar fields pass through
    assert "50 items" in content["summary"]

    # The current turn (last message) is preserved in full, even though it's large.
    current = out[1]
    assert isinstance(current, ModelRequest)
    last_part = current.parts[0]
    assert isinstance(last_part, ToolReturnPart)
    assert last_part.content == big


def test_summary_samples_the_first_non_empty_list() -> None:
    """A result with alternative lists (`rows` or `groups`) keeps the one that holds the answer."""
    rows = [{"id": str(i), "content": "x" * 100} for i in range(50)]
    big = {"groups": [], "rows": rows, "total": 50}
    prior = ModelRequest(parts=[ToolReturnPart(tool_name="list_notes", content=big, tool_call_id="c1")])
    last = ModelRequest(parts=[UserPromptPart(content="next")])

    out = compact_prior_tool_results([prior, last], SUMMARIZABLE)

    summarized = out[0]
    assert isinstance(summarized, ModelRequest)
    part = summarized.parts[0]
    assert isinstance(part, ToolReturnPart)
    content = cast(dict[str, Any], part.content)
    assert len(content["rows"]) == 5
    assert "groups" not in content
    assert "50 items in 'rows'" in content["summary"]


def test_rebuilt_message_keeps_identity_fields() -> None:
    prior = ModelRequest(
        parts=[ToolReturnPart(tool_name="list_notes", content=_notes_payload(50), tool_call_id="c1")],
        run_id="run-1",
    )
    last = ModelRequest(parts=[UserPromptPart(content="next")])

    out = compact_prior_tool_results([prior, last], SUMMARIZABLE)

    rebuilt = out[0]
    assert isinstance(rebuilt, ModelRequest)
    assert rebuilt is not prior
    assert rebuilt.run_id == "run-1"
    assert rebuilt.timestamp == prior.timestamp
    assert isinstance(rebuilt.parts[0], ToolReturnPart)
    assert rebuilt.parts[0].tool_call_id == "c1"


def test_non_summarizable_tool_is_untouched() -> None:
    prior = ModelRequest(parts=[ToolReturnPart(tool_name="other_tool", content=_notes_payload(50), tool_call_id="c1")])
    last = ModelRequest(parts=[ToolReturnPart(tool_name="list_notes", content={}, tool_call_id="c2")])

    out = compact_prior_tool_results([prior, last], SUMMARIZABLE)

    assert out[0] is prior


def test_small_result_is_untouched() -> None:
    prior = ModelRequest(parts=[ToolReturnPart(tool_name="list_notes", content=_notes_payload(1), tool_call_id="c1")])
    last = ModelRequest(parts=[ToolReturnPart(tool_name="list_notes", content={}, tool_call_id="c2")])

    out = compact_prior_tool_results([prior, last], SUMMARIZABLE)

    assert out[0] is prior  # below LARGE_RESULT_THRESHOLD


def test_single_message_returned_unchanged() -> None:
    msgs: list[ModelMessage] = [
        ModelRequest(parts=[ToolReturnPart(tool_name="list_notes", content=_notes_payload(50), tool_call_id="c1")])
    ]
    out = compact_prior_tool_results(msgs, SUMMARIZABLE)
    assert out == msgs


CONTEXT_WINDOW = 10_000


def _capturing_model(received: list[list[ModelMessage]]) -> FunctionModel:
    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        received.append(messages)
        return ModelResponse(parts=[TextPart(content="ok")])

    return FunctionModel(capture, profile={"context_window": CONTEXT_WINDOW})


def _history_with_big_result(last_response_tokens: int | None) -> list[ModelMessage]:
    """A prior turn that listed notes; its final response carries the usage that sets `context_window_used`."""
    usage = RequestUsage(input_tokens=last_response_tokens, output_tokens=0) if last_response_tokens else RequestUsage()
    return [
        ModelRequest(parts=[UserPromptPart(content="list my notes")]),
        ModelResponse(parts=[ToolCallPart(tool_name="list_notes", args={}, tool_call_id="c1")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="list_notes", content=_notes_payload(50), tool_call_id="c1")]),
        ModelResponse(parts=[TextPart(content="here they are")], usage=usage),
    ]


def _tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]


async def _run_with_history(history: list[ModelMessage]) -> list[ModelMessage]:
    """End to end: what the model actually receives after the processor ran."""
    received: list[list[ModelMessage]] = []
    agent = Agent(
        _capturing_model(received),
        capabilities=[ProcessHistory(build_tool_result_summarizer(SUMMARIZABLE))],
    )
    await agent.run("thanks", message_history=history)
    return received[0]


async def test_model_receives_compacted_prior_results_when_window_is_filling() -> None:
    over_threshold = int(CONTEXT_WINDOW * COMPACT_ABOVE_WINDOW_FRACTION) + 1
    received = await _run_with_history(_history_with_big_result(over_threshold))

    tool_returns = _tool_returns(received)
    assert len(tool_returns) == 1
    assert cast(dict[str, Any], tool_returns[0].content)["_summarized"] is True


async def test_prior_results_kept_verbatim_while_window_has_room() -> None:
    under_threshold = int(CONTEXT_WINDOW * COMPACT_ABOVE_WINDOW_FRACTION) - 1
    received = await _run_with_history(_history_with_big_result(under_threshold))

    tool_returns = _tool_returns(received)
    assert len(tool_returns) == 1
    assert tool_returns[0].content == _notes_payload(50)


async def test_compaction_stays_on_after_a_compacted_turn_reports_low_usage() -> None:
    """Storage keeps the full history, so a low-usage reply to a compacted request must not turn compaction off."""
    over_threshold = int(CONTEXT_WINDOW * COMPACT_ABOVE_WINDOW_FRACTION) + 1
    under_threshold = int(CONTEXT_WINDOW * COMPACT_ABOVE_WINDOW_FRACTION) - 1
    history = [
        *_history_with_big_result(over_threshold),
        ModelRequest(parts=[UserPromptPart(content="thanks")]),
        ModelResponse(parts=[TextPart(content="ok")], usage=RequestUsage(input_tokens=under_threshold)),
    ]
    received = await _run_with_history(history)

    assert cast(dict[str, Any], _tool_returns(received)[0].content)["_summarized"] is True


async def test_unknown_window_usage_still_compacts() -> None:
    """No usage on any response -> window fraction is unknown -> compact."""
    received = await _run_with_history(_history_with_big_result(None))

    assert cast(dict[str, Any], _tool_returns(received)[0].content)["_summarized"] is True
