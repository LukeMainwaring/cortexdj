"""The named-agent registry: the one place the runtime hands out agents by name."""

import pytest
from pydantic_ai.messages import InstructionPart, ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from cortexdj.agents.assistant import SYSTEM_PROMPT, build_assistant
from cortexdj.agents.registry import ASSISTANT, THREAD_TITLE, UnknownAgent, get_agent
from cortexdj.core.config import get_settings
from tests.factories import make_deps


def test_unknown_name_raises_unknown_agent() -> None:
    with pytest.raises(UnknownAgent, match="no-such-agent"):
        get_agent("no-such-agent")


def test_lookup_returns_the_same_built_agent_each_time() -> None:
    assert get_agent(ASSISTANT) is get_agent(ASSISTANT)
    assert get_agent(THREAD_TITLE) is get_agent(THREAD_TITLE)


async def test_assistant_runs_with_test_model() -> None:
    agent = get_agent(ASSISTANT)
    assert agent.name == ASSISTANT  # labels the Logfire run span
    with agent.override(model=TestModel(call_tools=[])):
        result = await agent.run("hello", deps=make_deps())
    assert result.output


def test_the_assistant_is_built_on_the_agent_model_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    # Built here, not through get_agent: the registry caches the agent for the process.
    monkeypatch.setenv("AGENT_MODEL", "anthropic:claude-haiku-5-5")
    get_settings.cache_clear()
    try:
        model = build_assistant().model
    finally:
        get_settings.cache_clear()
    assert isinstance(model, Model)
    assert model.model_name == "claude-haiku-5-5"


def test_assistant_caches_the_prompt_prefix_and_the_conversation() -> None:
    settings = get_agent(ASSISTANT).root_capability.get_model_settings()
    assert isinstance(settings, dict)
    assert settings.get("cache") is True


async def test_the_system_prompt_reaches_the_model_as_a_static_instruction() -> None:
    """The provider caches up to the last static instruction; rules inside a callable would miss it."""
    seen: list[InstructionPart] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(info.model_request_parameters.instruction_parts or [])
        return ModelResponse(parts=[TextPart("ok")])

    agent = get_agent(ASSISTANT)
    with agent.override(model=FunctionModel(respond)):
        await agent.run("Hello", deps=make_deps())

    static = [p.content for p in seen if not p.dynamic]
    assert any(SYSTEM_PROMPT in text for text in static)


def test_the_system_prompt_quadrant_list_makes_no_band_claims() -> None:
    quadrant_lines = [line for line in SYSTEM_PROMPT.splitlines() if line.startswith("- **")]

    assert len(quadrant_lines) == 4
    for line in quadrant_lines:
        assert not [band for band in ("theta", "alpha", "beta", "gamma") if band in line.lower()]


def test_the_system_prompt_band_line_lists_the_four_recipe_bands() -> None:
    assert "theta 4-8 Hz, alpha 8-13 Hz, beta 13-30 Hz, gamma 30-45 Hz" in SYSTEM_PROMPT
    assert "delta" not in SYSTEM_PROMPT.lower()
