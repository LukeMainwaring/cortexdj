"""Unit tests for utils.message_serialization round-trips."""

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.ui.vercel_ai.request_types import UIMessage

from cortexdj.utils.message_serialization import (
    deserialize_messages,
    dump_messages_for_frontend,
    extract_latest_user_text,
    prepare_messages_for_storage,
)


def _sample_messages() -> list[ModelRequest | ModelResponse]:
    return [
        ModelRequest(parts=[UserPromptPart(content="How was my session?")]),
        ModelResponse(parts=[TextPart(content="Mostly relaxed.")]),
    ]


class TestStorageRoundTrip:
    def test_dump_is_json_safe(self) -> None:
        dumped = prepare_messages_for_storage(list(_sample_messages()))
        assert all(isinstance(m, dict) for m in dumped)

    def test_deserialize_inverts_prepare(self) -> None:
        original = list(_sample_messages())
        restored = deserialize_messages(prepare_messages_for_storage(original))
        assert len(restored) == 2
        assert isinstance(restored[0], ModelRequest)
        assert isinstance(restored[1], ModelResponse)
        assert restored[1].parts[0].content == "Mostly relaxed."  # type: ignore[union-attr]


class TestDumpForFrontend:
    def test_produces_ui_roles_and_text(self) -> None:
        stored = prepare_messages_for_storage(list(_sample_messages()))
        ui = dump_messages_for_frontend(stored)
        assert [m["role"] for m in ui] == ["user", "assistant"]
        assistant_parts = ui[1]["parts"]
        assert any(p.get("type") == "text" and p.get("text") == "Mostly relaxed." for p in assistant_parts)


class TestExtractLatestUserText:
    def test_returns_last_user_text(self) -> None:
        messages = [
            UIMessage.model_validate({"id": "1", "role": "user", "parts": [{"type": "text", "text": "first"}]}),
            UIMessage.model_validate({"id": "2", "role": "assistant", "parts": [{"type": "text", "text": "reply"}]}),
            UIMessage.model_validate({"id": "3", "role": "user", "parts": [{"type": "text", "text": "second"}]}),
        ]
        assert extract_latest_user_text(messages) == "second"

    def test_empty_history_returns_empty_string(self) -> None:
        assert extract_latest_user_text([]) == ""
