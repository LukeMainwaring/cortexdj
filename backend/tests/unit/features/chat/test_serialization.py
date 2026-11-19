"""Message serialization: storage round-trip, frontend dump, latest-user-text."""

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.ui.vercel_ai.request_types import TextUIPart, UIMessage

from cortexdj.features.chat.serialization import (
    deserialize_messages,
    dump_messages_for_frontend,
    extract_latest_user_text,
    prepare_messages_for_storage,
)


def _conversation() -> list[ModelRequest | ModelResponse]:
    return [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="hi there")]),
    ]


def test_storage_round_trip_preserves_user_prompt() -> None:
    dumped = prepare_messages_for_storage(list(_conversation()))
    assert isinstance(dumped, list)
    assert isinstance(dumped[0], dict)

    restored = deserialize_messages(dumped)
    assert len(restored) == 2
    request = restored[0]
    assert isinstance(request, ModelRequest)
    assert any(isinstance(p, UserPromptPart) and p.content == "hello" for p in request.parts)


def test_dump_messages_for_frontend_emits_vercel_shape() -> None:
    stored = prepare_messages_for_storage(list(_conversation()))
    ui_messages = dump_messages_for_frontend(stored)

    # The frontend reads each message's id, role, and parts. The ids are derived by pydantic-ai
    # and need only be distinct; its metadata is pydantic-ai's own and not asserted.
    assert len({m["id"] for m in ui_messages}) == 2
    assert [{"role": m["role"], "parts": m["parts"]} for m in ui_messages] == [
        {"role": "user", "parts": [{"type": "text", "text": "hello", "state": "done"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "hi there", "state": "done"}]},
    ]


def test_extract_latest_user_text_returns_last_user_text() -> None:
    messages = [
        UIMessage(id="1", role="user", parts=[TextUIPart(text="first")]),
        UIMessage(id="2", role="assistant", parts=[TextUIPart(text="reply")]),
        UIMessage(id="3", role="user", parts=[TextUIPart(text="second")]),
    ]
    assert extract_latest_user_text(messages) == "second"


def test_extract_latest_user_text_empty_when_no_user_message() -> None:
    assert extract_latest_user_text([]) == ""
    only_assistant = [UIMessage(id="1", role="assistant", parts=[TextUIPart(text="hi")])]
    assert extract_latest_user_text(only_assistant) == ""
