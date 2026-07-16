"""Unit tests for services.title_generator.

The generator runs as a fire-and-forget background task on its own session
(outside the request's transaction), so it's covered here with a patched
OpenAI client and session context rather than at the integration tier.
"""

import contextlib
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cortexdj.services.title_generator import _create_fallback_title, generate_thread_title

pytestmark = pytest.mark.anyio


class TestCreateFallbackTitle:
    def test_short_message_passes_through(self) -> None:
        assert _create_fallback_title("Analyze my last session") == "Analyze my last session"

    def test_long_message_truncates_at_word_boundary(self) -> None:
        message = "Please analyze my most recent relaxation session and compare it with last week"
        title = _create_fallback_title(message)
        assert title.endswith("...")
        assert len(title) <= 43  # 40 chars + ellipsis
        # No mid-word cut: everything before "..." is a whole-word prefix.
        assert message.startswith(title.removesuffix("..."))

    def test_strips_surrounding_whitespace(self) -> None:
        assert _create_fallback_title("  hi  ") == "hi"


@contextlib.asynccontextmanager
async def _fake_session() -> AsyncIterator[MagicMock]:
    yield MagicMock()


def _openai_client_returning(text: str) -> MagicMock:
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=MagicMock(output_text=text))
    return client


class TestGenerateThreadTitle:
    async def test_saves_llm_title_with_quotes_stripped(self) -> None:
        with (
            patch(
                "cortexdj.services.title_generator.AsyncOpenAI",
                return_value=_openai_client_returning('"Neural Beats Recap"'),
            ),
            patch("cortexdj.services.title_generator.get_async_sqlalchemy_session", _fake_session),
            patch("cortexdj.services.title_generator.Thread.update_title", new=AsyncMock()) as update_title,
        ):
            await generate_thread_title("t-1", "chat", "user msg", "assistant msg")

        assert update_title.call_args.args[1:] == ("t-1", "chat", "Neural Beats Recap")

    async def test_empty_llm_output_saves_nothing(self) -> None:
        with (
            patch(
                "cortexdj.services.title_generator.AsyncOpenAI",
                return_value=_openai_client_returning(""),
            ),
            patch("cortexdj.services.title_generator.get_async_sqlalchemy_session", _fake_session),
            patch("cortexdj.services.title_generator.Thread.update_title", new=AsyncMock()) as update_title,
        ):
            await generate_thread_title("t-1", "chat", "user msg", "assistant msg")

        update_title.assert_not_called()

    async def test_llm_failure_falls_back_to_truncated_user_message(self) -> None:
        failing_client = MagicMock()
        failing_client.responses.create = AsyncMock(side_effect=RuntimeError("api down"))
        with (
            patch("cortexdj.services.title_generator.AsyncOpenAI", return_value=failing_client),
            patch("cortexdj.services.title_generator.get_async_sqlalchemy_session", _fake_session),
            patch("cortexdj.services.title_generator.Thread.update_title", new=AsyncMock()) as update_title,
        ):
            await generate_thread_title("t-1", "chat", "Compare my sessions", "resp")

        assert update_title.call_args.args[1:] == ("t-1", "chat", "Compare my sessions")

    async def test_fallback_save_failure_is_swallowed(self) -> None:
        # A background task must never propagate — it has no request to fail.
        failing_client = MagicMock()
        failing_client.responses.create = AsyncMock(side_effect=RuntimeError("api down"))
        with (
            patch("cortexdj.services.title_generator.AsyncOpenAI", return_value=failing_client),
            patch("cortexdj.services.title_generator.get_async_sqlalchemy_session", _fake_session),
            patch(
                "cortexdj.services.title_generator.Thread.update_title",
                new=AsyncMock(side_effect=RuntimeError("db down")),
            ),
        ):
            await generate_thread_title("t-1", "chat", "user msg", "resp")
