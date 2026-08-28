"""History processor: compact large prior-turn tool results, change nothing else.

The point of these tests is the *negative* half of the contract. Summarization
rebuilds a ``ToolReturnPart`` and its ``ModelRequest``, and a hand-written
constructor call silently drops every field it forgets to copy — including
fields a future Pydantic AI adds. Each rebuild case therefore carries non-default
sentinel values on the fields we are not allowed to touch.
"""

from typing import Any, cast

import pytest
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

from cortexdj.agents.history_processor import summarize_tool_results


def _tracks_payload(n: int) -> dict[str, Any]:
    """Comfortably over LARGE_RESULT_THRESHOLD once serialized."""
    return {
        "tracks": [{"id": str(i), "name": f"t{i}", "artist": "x" * 100} for i in range(n)],
        "total_available": n,
    }


def _large_return(**overrides: Any) -> ToolReturnPart:
    return ToolReturnPart(tool_name="search_tracks", content=_tracks_payload(50), tool_call_id="c1", **overrides)


def _trailing_turn() -> ModelRequest:
    """A current turn the processor must leave alone, keeping the interesting message prior."""
    return ModelRequest(parts=[UserPromptPart(content="next")])


class TestSummarization:
    def test_large_prior_result_is_compacted(self) -> None:
        prior = ModelRequest(parts=[_large_return()])

        out = summarize_tool_results([prior, _trailing_turn()])

        assert len(out) == 2, "parts are rewritten; messages are never added or dropped"
        part = cast(ModelRequest, out[0]).parts[0]
        assert isinstance(part, ToolReturnPart)
        content = cast(dict[str, Any], part.content)
        assert content["_summarized"] is True
        assert content["tool"] == "search_tracks"
        assert content["count"] == 50
        assert len(content["sample"]) == 5
        assert content["total_available"] == 50

    def test_current_turn_is_never_compacted(self) -> None:
        big = ModelRequest(parts=[_large_return()])

        out = summarize_tool_results([ModelRequest(parts=[UserPromptPart(content="hi")]), big])

        assert out[1] is big

    def test_already_summarized_result_is_untouched(self) -> None:
        prior = ModelRequest(
            parts=[ToolReturnPart(tool_name="search_tracks", content={"_summarized": True}, tool_call_id="c1")]
        )
        assert summarize_tool_results([prior, _trailing_turn()])[0] is prior

    def test_below_threshold_result_is_untouched(self) -> None:
        prior = ModelRequest(
            parts=[ToolReturnPart(tool_name="search_tracks", content=_tracks_payload(1), tool_call_id="c1")]
        )
        assert summarize_tool_results([prior, _trailing_turn()])[0] is prior

    def test_non_summarizable_tool_is_untouched(self) -> None:
        prior = ModelRequest(
            parts=[ToolReturnPart(tool_name="get_track_info", content=_tracks_payload(50), tool_call_id="c1")]
        )
        assert summarize_tool_results([prior, _trailing_turn()])[0] is prior

    def test_single_message_is_returned_unchanged(self) -> None:
        messages: list[ModelMessage] = [ModelRequest(parts=[_large_return()])]
        assert summarize_tool_results(messages) == messages


class TestFieldPreservation:
    """Only ``content`` may change; the sentinels below must survive the rebuild."""

    def test_tool_return_part_keeps_every_other_field(self) -> None:
        original = _large_return(metadata={"trace": "sentinel"})
        out = summarize_tool_results([ModelRequest(parts=[original]), _trailing_turn()])

        rebuilt = cast(ModelRequest, out[0]).parts[0]
        assert isinstance(rebuilt, ToolReturnPart)
        assert rebuilt is not original
        assert rebuilt.content != original.content, "content is the one field we do rewrite"

        # Everything else, compared field-by-field so a field added by a future
        # Pydantic AI release is covered without this test naming it.
        changed = {
            name: (getattr(original, name), getattr(rebuilt, name))
            for name in original.__dataclass_fields__
            if getattr(original, name) != getattr(rebuilt, name)
        }
        assert set(changed) == {"content"}, f"unexpectedly rewritten fields: {sorted(set(changed) - {'content'})}"
        assert rebuilt.metadata == {"trace": "sentinel"}

    def test_model_request_keeps_every_other_field(self) -> None:
        original = ModelRequest(
            parts=[_large_return()],
            instructions="sentinel instructions",
            run_id="run-sentinel",
            conversation_id="conv-sentinel",
        )
        out = summarize_tool_results([original, _trailing_turn()])

        rebuilt = out[0]
        assert isinstance(rebuilt, ModelRequest)
        assert rebuilt is not original

        changed = {
            name
            for name in original.__dataclass_fields__
            if getattr(original, name) != getattr(rebuilt, name) and name != "parts"
        }
        assert changed == set(), f"unexpectedly rewritten fields: {sorted(changed)}"
        assert rebuilt.instructions == "sentinel instructions"
        assert rebuilt.run_id == "run-sentinel"
        assert rebuilt.conversation_id == "conv-sentinel"
        assert rebuilt.timestamp == original.timestamp


@pytest.mark.anyio
async def test_model_receives_compacted_prior_results() -> None:
    """End to end: what the model is actually handed has the prior result compacted."""
    received: list[list[ModelMessage]] = []

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        received.append(messages)
        return ModelResponse(parts=[TextPart(content="ok")])

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="find me something")]),
        ModelResponse(parts=[ToolCallPart(tool_name="search_tracks", args={}, tool_call_id="c1")]),
        ModelRequest(parts=[_large_return()]),
        ModelResponse(parts=[TextPart(content="here they are")]),
    ]
    agent = Agent(FunctionModel(capture), capabilities=[ProcessHistory(summarize_tool_results)])
    await agent.run("thanks", message_history=history)

    tool_returns = [p for m in received[0] for p in m.parts if isinstance(p, ToolReturnPart)]
    assert len(tool_returns) == 1
    assert cast(dict[str, Any], tool_returns[0].content)["_summarized"] is True
