"""The chat adapter's sanitizing: what the route stores is what the run gets, with client compaction parts gone.

Pure, no DB and no provider: ``sanitize_messages`` is a function of the adapter's settings and the
messages, so an adapter over a ``TestModel`` agent is the real seam.
"""

import json
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

from cortexdj.agents.deps import AgentDeps
from cortexdj.features.chat.stream import _ChatAdapter
from tests.factories import make_chat_request


def _adapter() -> _ChatAdapter:
    agent: Agent[AgentDeps, Any] = Agent(TestModel(), deps_type=AgentDeps)
    body = json.dumps(make_chat_request("t-1", "hello")).encode()
    return _ChatAdapter(agent=agent, run_input=_ChatAdapter.build_run_input(body), sdk_version=7)


def _parts(messages: list[ModelMessage]) -> list[list[type[object]]]:
    return [[type(part) for part in message.parts] for message in messages]


def test_a_client_compaction_part_is_dropped_before_the_turn_is_stored() -> None:
    """A compaction part the client sent would trim the trusted history out from under the model."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="earlier")]),
        ModelResponse(parts=[CompactionPart(content="everything so far"), TextPart(content="ok")]),
        ModelRequest(parts=[UserPromptPart(content="now")]),
    ]

    sanitized = _adapter().sanitize_messages(messages)

    assert _parts(sanitized) == [[UserPromptPart], [TextPart], [UserPromptPart]]


def test_a_response_that_was_only_a_compaction_part_disappears_whole() -> None:
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="earlier")]),
        ModelResponse(parts=[CompactionPart(content="everything so far")]),
        ModelRequest(parts=[UserPromptPart(content="now")]),
    ]

    sanitized = _adapter().sanitize_messages(messages)

    assert _parts(sanitized) == [[UserPromptPart], [UserPromptPart]]


def test_an_approved_trailing_tool_call_survives_when_the_stored_copy_is_sanitized_with_the_approvals() -> None:
    """The run keeps a tool call the client just approved; the stored copy must keep it too, or they diverge."""
    body = {
        "trigger": "submit-message",
        "id": "t-1",
        "messages": [
            {"id": "u1", "role": "user", "parts": [{"type": "text", "text": "delete it"}]},
            {
                "id": "a1",
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool-delete_note",
                        "toolCallId": "call-1",
                        "state": "approval-responded",
                        "input": {"note_id": 7},
                        "approval": {"id": "appr-1", "approved": True},
                    }
                ],
            },
        ],
    }
    agent: Agent[AgentDeps, Any] = Agent(TestModel(), deps_type=AgentDeps)
    adapter = _ChatAdapter(
        agent=agent, run_input=_ChatAdapter.build_run_input(json.dumps(body).encode()), sdk_version=7
    )
    assert adapter.deferred_tool_results is not None

    with_approvals = adapter.sanitize_messages(adapter.messages, deferred_tool_results=adapter.deferred_tool_results)
    # The contrast: sanitized without them, the call is a dangling tail and goes, with a warning.
    with pytest.warns(UserWarning, match="unresolved tool call"):
        without = adapter.sanitize_messages(adapter.messages)

    assert _parts(with_approvals) == [[UserPromptPart], [ToolCallPart]]
    assert _parts(without) == [[UserPromptPart]]
