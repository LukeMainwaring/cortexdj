"""History processor for compacting large tool results in prior turns.

Large list-returning tool results stay in the conversation and are re-sent on
every request. This processor keeps the current turn intact (the model needs
the full result to reason about it) and replaces oversized results in earlier
turns with a compact summary: the scalar fields plus a small sample of the list.

Compaction is lossy, so it only runs once the history has actually gotten big:
the gate is the *peak* fraction of the model's window any response in the
history reported (read from the stored usage, so it carries across requests).
Below ``COMPACT_ABOVE_WINDOW_FRACTION`` the history is passed through
untouched. When the fraction is unknown (no response yet, or a model whose
window size isn't known) compaction runs — unknown is not the same as
empty, and re-sending a huge result is the worse failure.

Do not gate on ``ctx.context_window_used`` (the *latest* response): compaction
isn't persisted, so the response to a compacted request reports low usage while
the next request reloads the full history from storage. Gating on it switches
compaction off every other turn and can overflow the window.

Invariant: it rewrites parts in place and never adds or removes messages, so what
the run sends to the model lines up one-for-one with the history it was handed.
Compact an oversized result; do not drop the message carrying it.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ToolReturnPart,
)

from cortexdj.agents.deps import AgentDeps

logger = logging.getLogger(__name__)

LARGE_RESULT_THRESHOLD = 2000

SAMPLE_SIZE = 5

# Fraction of the model's context window in use before prior-turn results get
# compacted. Below this there is room to keep them verbatim for better recall.
COMPACT_ABOVE_WINDOW_FRACTION = 0.5


def _get_content_size(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    try:
        return len(json.dumps(content))
    except TypeError, ValueError:
        return 0


def _summarize_result(tool_name: str, content: dict[str, Any]) -> dict[str, Any]:
    """Keep the result's scalar fields (counts, totals, query) and a sample of its first non-empty list.

    Non-empty first because a result may carry alternative lists of which only one is filled
    (a query tool answering with ``rows`` or ``groups``); sampling an empty one would drop the answer.
    """
    summary: dict[str, Any] = {k: v for k, v in content.items() if not isinstance(v, (list, dict))}
    summary["_summarized"] = True
    summary["tool"] = tool_name

    lists = [k for k, v in content.items() if isinstance(v, list)]
    list_key = next((k for k in lists if content[k]), lists[0] if lists else None)
    if list_key is None:
        summary["summary"] = f"Large result from {tool_name}"
        summary["original_keys"] = list(content.keys())
        return summary

    items = content[list_key]
    summary["summary"] = f"{len(items)} items in '{list_key}', showing the first {min(len(items), SAMPLE_SIZE)}"
    summary[list_key] = items[:SAMPLE_SIZE]
    return summary


def _compact_tool_return_part(part: ToolReturnPart, summarizable_tools: frozenset[str]) -> ToolReturnPart:
    if part.tool_name not in summarizable_tools:
        return part

    content = part.content
    if not isinstance(content, dict) or content.get("_summarized"):
        return part

    size = _get_content_size(content)
    if size < LARGE_RESULT_THRESHOLD:
        return part

    summarized = _summarize_result(part.tool_name, content)
    logger.info(f"Summarized {part.tool_name} result: {size} chars -> {_get_content_size(summarized)} chars")
    # `replace` keeps every other field (timestamp, metadata, outcome, ...) intact.
    return replace(part, content=summarized)


def _peak_window_used(messages: list[ModelMessage], context_window: int | None) -> float | None:
    """Largest fraction of the window any response reported; None when the window or usage is unknown."""
    if not context_window:
        return None
    peak = max((m.usage.total_tokens for m in messages if isinstance(m, ModelResponse)), default=0)
    return peak / context_window if peak else None


def compact_prior_tool_results(messages: list[ModelMessage], summarizable_tools: frozenset[str]) -> list[ModelMessage]:
    """Compact large results from the given tools in prior messages; the current turn is preserved in full."""
    if len(messages) <= 1:
        return messages

    processed_messages: list[ModelMessage] = []

    for i, message in enumerate(messages):
        is_current_turn = i == len(messages) - 1

        if is_current_turn or not isinstance(message, ModelRequest):
            processed_messages.append(message)
            continue

        new_parts: list[ModelRequestPart] = []
        modified = False
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                processed_part = _compact_tool_return_part(part, summarizable_tools)
                modified = modified or processed_part is not part
                new_parts.append(processed_part)
            else:
                new_parts.append(part)

        # `replace` keeps run_id, conversation_id, timestamp, and metadata on the message.
        processed_messages.append(replace(message, parts=new_parts) if modified else message)

    return processed_messages


def build_tool_result_summarizer(
    summarizable_tools: frozenset[str],
) -> Callable[[RunContext[AgentDeps], list[ModelMessage]], list[ModelMessage]]:
    """Build a history processor that compacts large results from the given tools once the window fills.

    The summarizable tool names come from the enabled capabilities (each declares
    its own ``summarizable_tools``), so the shared runtime carries no
    feature-specific tool names. Wired in agents/assistant.py.
    """

    def summarize_tool_results(ctx: RunContext[AgentDeps], messages: list[ModelMessage]) -> list[ModelMessage]:
        used = _peak_window_used(messages, ctx.model.context_window)
        if used is not None and used < COMPACT_ABOVE_WINDOW_FRACTION:
            return messages
        return compact_prior_tool_results(messages, summarizable_tools)

    return summarize_tool_results
