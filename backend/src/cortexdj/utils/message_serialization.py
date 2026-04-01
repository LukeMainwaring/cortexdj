from typing import Any

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.ui.vercel_ai.request_types import TextUIPart, UIMessage


def prepare_messages_for_storage(
    messages: list[ModelMessage],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    return result


def deserialize_messages(message_data: list[dict[str, Any]]) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_python(message_data)


def dump_messages_for_frontend(message_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_messages = deserialize_messages(message_data)
    ui_messages = VercelAIAdapter.dump_messages(model_messages)
    return [msg.model_dump(mode="json", by_alias=True) for msg in ui_messages]


def extract_latest_user_text(messages: list[UIMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            for part in msg.parts:
                if isinstance(part, TextUIPart):
                    return part.text
    return ""
