"""Fallback title: verbatim when short, word-boundary truncation when long."""

from cortexdj.features.chat.title_generator import _create_fallback_title


def test_short_message_used_verbatim_and_stripped() -> None:
    assert _create_fallback_title("  Buy milk  ") == "Buy milk"


def test_long_message_truncated_at_word_boundary() -> None:
    title = _create_fallback_title("Can you help me plan a budget for the trip to Lisbon next spring?")
    assert title == "Can you help me plan a budget for the..."
