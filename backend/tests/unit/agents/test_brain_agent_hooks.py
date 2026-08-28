"""Unit tests for the brain_agent hooks module.

An unexpected exception out of a tool body must become a *native* failed tool
outcome — the run survives, the model sees the failure, and the tool's retry
budget is untouched. Asserted end-to-end through a ``FunctionModel``-driven
agent rather than by calling the hook directly: the failed-outcome bookkeeping
lives in Pydantic AI's tool manager, so a direct call would prove nothing about
what the model actually receives.
"""

from typing import cast

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.agents.deps import AgentDeps
from cortexdj.agents.hooks import build_brain_agent_hooks


def _call_explode_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    last_part = messages[-1].parts[-1]
    if isinstance(last_part, ToolReturnPart):
        return ModelResponse(parts=[TextPart(content=f"Sorry: {last_part.content}")])
    return ModelResponse(parts=[ToolCallPart(tool_name="explode", args={})])


def _agent_with_failing_tool(error: Exception) -> Agent[AgentDeps]:
    agent: Agent[AgentDeps] = Agent(
        FunctionModel(_call_explode_then_answer),
        deps_type=AgentDeps,
        capabilities=[build_brain_agent_hooks()],
    )

    @agent.tool_plain
    def explode() -> str:
        raise error

    return agent


def _no_db_deps() -> AgentDeps:
    return AgentDeps(db=cast(AsyncSession, None))


class TestUnexpectedToolException:
    @pytest.mark.anyio
    async def test_becomes_a_native_failed_tool_result(self) -> None:
        result = await _agent_with_failing_tool(RuntimeError("spotify 500")).run("go", deps=_no_db_deps())

        returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        assert len(returns) == 1, "one failed call, one result — ToolFailed must not spend the retry budget"
        assert returns[0].outcome == "failed"
        assert result.output.startswith("Sorry"), "the run completed; nothing escaped to kill the stream"

    @pytest.mark.anyio
    async def test_message_names_the_tool_and_exception_class(self) -> None:
        result = await _agent_with_failing_tool(RuntimeError("spotify 500")).run("go", deps=_no_db_deps())

        content = str(next(p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)).content)
        assert "explode" in content
        assert "RuntimeError" in content

    @pytest.mark.anyio
    async def test_exception_text_never_reaches_the_model(self) -> None:
        # An exception message can carry connection strings, tokens, or upstream
        # response bodies — see docs/adr/0003-two-layer-tool-error-convention.md.
        secret = "postgres://user:hunter2@db.internal:5432"
        result = await _agent_with_failing_tool(ConnectionError(secret)).run("go", deps=_no_db_deps())

        content = str(next(p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)).content)
        assert secret not in content
        assert "hunter2" not in content
        assert "ConnectionError" in content


class TestBuildBrainAgentHooks:
    def test_returns_hooks_instance(self) -> None:
        assert isinstance(build_brain_agent_hooks(), Hooks)
