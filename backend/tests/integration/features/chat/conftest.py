"""Chat-slice fakes for the integration tier: title generation never runs, and the model stream doubles.

Title generation opens its OWN session (a detached asyncio task) and runs a model, so it
is out of the per-test rollback: a real run would write to the test database for good.
The chat route takes it as a dependency, so it is replaced here through
``app.dependency_overrides`` (docs/adr/0001-test-tiers-and-fakes.md). It lives in this
slice's conftest rather than the shared ``client`` fixture so that
``tests/integration/conftest.py`` never imports feature code and survives every prune.

The stream doubles stand in for the assistant's model through
``get_agent(ASSISTANT).override(model=FunctionModel(stream_function=...))``. They are
stream functions because ``/agent/chat`` only ever streams, so a plain (non-streaming)
``FunctionModel`` would never be called. Import them from this module.
"""

from collections.abc import AsyncIterator, Iterator

import pytest
from pydantic_ai.messages import ModelMessage, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from cortexdj.app import app
from cortexdj.features.chat.title_generator import TitleGenerator, get_title_generator

CHAT_URL = "/api/agent/chat"


@pytest.fixture(autouse=True)
def _no_title_generation() -> Iterator[None]:
    async def _no_title(*, thread_id: str, user_message: str, assistant_response: str) -> None:
        return None

    def _title_override() -> TitleGenerator:
        return _no_title

    app.dependency_overrides[get_title_generator] = _title_override
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_title_generator, None)


async def reply_ok(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    """A model that answers every turn with "ok"."""
    yield "ok"


async def die_mid_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    """A provider that fails after the reply has started, so the run never completes."""
    yield "partial"
    raise RuntimeError("provider is down")


class PromptRecorder:
    """A model that answers "ok" and keeps the user prompts of every request it was sent, one list per request."""

    def __init__(self) -> None:
        self.prompts: list[list[object]] = []

    async def _reply(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        self.prompts.append([part.content for m in messages for part in m.parts if isinstance(part, UserPromptPart)])
        yield "ok"

    def model(self) -> FunctionModel:
        return FunctionModel(stream_function=self._reply)
