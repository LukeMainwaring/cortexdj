"""History processor for summarizing large tool results.

Prevents token bloat from large Spotify and EEG tool responses by summarizing
tool results in historical messages. The current turn sees full results for
accurate reasoning, while subsequent turns see compact summaries.
"""

import json
import logging
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ToolReturnPart,
)

logger = logging.getLogger(__name__)

LARGE_RESULT_THRESHOLD = 2000

SAMPLE_SIZE = 5

SUMMARIZABLE_TOOLS = frozenset(
    {
        "find_relaxing_tracks",
        "build_mood_playlist",
        "get_listening_history",
        "search_tracks",
        "get_my_playlists",
        "get_my_saved_tracks",
    }
)


def _get_content_size(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    try:
        return len(json.dumps(content))
    except (TypeError, ValueError):
        return 0


def _summarize_list_result(
    tool_name: str,
    content: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "_summarized": True,
        "tool": tool_name,
    }

    if "playlists" in content and isinstance(content["playlists"], list):
        playlists = content["playlists"]
        summary["summary"] = f"Retrieved {len(playlists)} playlists"
        summary["count"] = len(playlists)
        summary["sample"] = playlists[:SAMPLE_SIZE]
        if "total" in content:
            summary["total_available"] = content["total"]

    elif "tracks" in content and isinstance(content["tracks"], list):
        tracks = content["tracks"]
        summary["summary"] = f"Retrieved {len(tracks)} tracks"
        summary["count"] = len(tracks)
        summary["sample"] = tracks[:SAMPLE_SIZE]
        if "total_available" in content:
            summary["total_available"] = content["total_available"]
        elif "total_results" in content:
            summary["total_available"] = content["total_results"]

    elif "saved_tracks" in content and isinstance(content["saved_tracks"], list):
        saved = content["saved_tracks"]
        summary["summary"] = f"Retrieved {len(saved)} saved tracks"
        summary["count"] = len(saved)
        summary["sample"] = saved[:SAMPLE_SIZE]
        if "total" in content:
            summary["total_available"] = content["total"]

    else:
        summary["summary"] = f"Large result from {tool_name}"
        summary["original_keys"] = list(content.keys()) if isinstance(content, dict) else None

    return summary


def _process_tool_return_part(part: ToolReturnPart) -> ToolReturnPart:
    if part.tool_name not in SUMMARIZABLE_TOOLS:
        return part

    content = part.content
    if content is None:
        return part

    if isinstance(content, dict) and content.get("_summarized"):
        return part

    size = _get_content_size(content)
    if size < LARGE_RESULT_THRESHOLD:
        return part

    if isinstance(content, dict):
        summarized = _summarize_list_result(part.tool_name, content)
        logger.info(f"Summarized {part.tool_name} result: {size} chars -> {_get_content_size(summarized)} chars")
        return ToolReturnPart(
            tool_name=part.tool_name,
            content=summarized,
            tool_call_id=part.tool_call_id,
            timestamp=part.timestamp,
        )

    return part


def summarize_tool_results(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Compact large tool results in prior messages; the current turn is preserved in full."""
    if len(messages) <= 1:
        return messages

    processed_messages: list[ModelMessage] = []

    for i, message in enumerate(messages):
        is_current_turn = i == len(messages) - 1

        if is_current_turn:
            processed_messages.append(message)
            continue

        if isinstance(message, ModelRequest):
            new_parts: list[ModelRequestPart] = []
            modified = False

            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    processed_part = _process_tool_return_part(part)
                    if processed_part is not part:
                        modified = True
                    new_parts.append(processed_part)
                else:
                    new_parts.append(part)

            if modified:
                processed_messages.append(
                    ModelRequest(
                        parts=new_parts,
                        instructions=message.instructions,
                    )
                )
            else:
                processed_messages.append(message)
        else:
            processed_messages.append(message)

    return processed_messages
