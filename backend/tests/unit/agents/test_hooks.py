"""Error recovery: an unexpected exception in a tool or in output processing becomes something the model can act on."""

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from cortexdj.agents.deps import AgentDeps
from cortexdj.agents.hooks import build_agent_hooks
from tests.factories import make_deps


def _call_explode_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    last_part = messages[-1].parts[-1]
    if isinstance(last_part, ToolReturnPart):
        return ModelResponse(parts=[TextPart(content=f"Sorry: {last_part.content}")])
    return ModelResponse(parts=[ToolCallPart(tool_name="explode", args={})])


def _answer_then_answer_again(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[TextPart(content="first")])
    # A retry of text output reaches the model as a user turn carrying the feedback.
    feedback = messages[-1].parts[-1]
    assert isinstance(feedback, UserPromptPart)
    return ModelResponse(parts=[TextPart(content=f"second after: {feedback.content}")])


async def test_unexpected_tool_exception_becomes_failed_result() -> None:
    agent: Agent[AgentDeps] = Agent(
        FunctionModel(_call_explode_then_answer), deps_type=AgentDeps, capabilities=[build_agent_hooks()]
    )

    @agent.tool_plain
    def explode() -> str:
        raise RuntimeError("boom")

    result = await agent.run("go", deps=make_deps())

    # The run completed (no exception escaped) and the failure is a proper failed tool result.
    returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
    assert len(returns) == 1
    assert returns[0].outcome == "failed"
    failure = (
        "The explode tool failed unexpectedly (RuntimeError). "
        "Tell the user briefly what you were trying to do and suggest they retry or rephrase."
    )
    assert returns[0].content == failure
    assert result.output == f"Sorry: {failure}"  # the model was shown the failure and answered


async def test_unexpected_output_validator_exception_becomes_retry() -> None:
    agent: Agent[AgentDeps] = Agent(
        FunctionModel(_answer_then_answer_again), deps_type=AgentDeps, capabilities=[build_agent_hooks()]
    )

    @agent.output_validator
    def explode_on_first(ctx: RunContext[AgentDeps], output: str) -> str:
        if output == "first":
            raise RuntimeError("boom")
        return output

    result = await agent.run("go", deps=make_deps())

    # The run completed, and the model was told what failed and answered again. Only the hook's
    # message is asserted: pydantic-ai wraps it in retry framing of its own.
    assert result.output.startswith("second after: ")
    assert (
        "Processing your final answer failed unexpectedly (RuntimeError). "
        "Answer again; if it fails again, tell the user briefly what you were trying to do."
    ) in result.output
