"""Run one eval case through the registered assistant, the way the chat stream runs a turn.

The task asks the registry for the assistant and runs the case's prompt through it, scrubbed
as the chat stream scrubs a turn, so an eval measures the agent that ships: its instructions,
capabilities, and model, on the text the model would see in production. The shipped
case reaches no database-backed tool, so the deps carry the mock session from
``tests.factories.make_deps``. ``run_assistant`` takes only a case's inputs, so a test whose
cases need stored data runs its own task instead: a closure over the ``eval_sessions``
fixture (``conftest.py``) that opens a session per case and passes it as
``make_deps(session)``. Seed what the cases need in ``backend/scripts/run-evals.sh`` after
the migrate step.

The task also records the run's tool calls, each paired with what the tool answered, as the
``tool_calls`` eval attribute. An evaluator over what a tool answered reads it from
``ctx.attributes``, the report prints it below the table, and a judge that sees the input is
shown the returns of its ``judged_tools`` beside the case's inputs (``JudgedTurn``, built in
``judges.py``), so a figure the reply takes from a tool is one the judge can find. How many
times a tool was called is counted from the run's spans instead (``MaxCallsOf``), where the
library decides what counts as a call.
"""

from collections.abc import Collection
from typing import Any

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ToolCallPart, ToolReturnPart
from pydantic_core import to_json, to_jsonable_python
from pydantic_evals import Case, set_eval_attribute

from cortexdj.agents.registry import ASSISTANT, get_agent
from tests.factories import make_deps


class AgentTurn(BaseModel):
    """A case's inputs: what the user says.

    Add here whatever a case needs in place before the turn (stored rows, facts from earlier
    chats). A judge that sees the input is shown the whole model as JSON.
    """

    prompt: str


class RecordedToolCall(BaseModel):
    """One tool call the run made: the arguments the model sent and what the tool answered.

    ``returned`` is the tool's answer as JSON, the retry message when the tool sent the call
    back (``outcome="retried"``), the error when it failed, and None when no answer came.
    """

    name: str
    args: dict[str, Any]
    returned: Any
    outcome: str = "success"


class JudgedToolReturn(BaseModel):
    name: str
    returned: Any


class JudgedTurn(AgentTurn):
    """What a judge that sees the input is shown: the case's inputs plus what chosen tools returned (``judges.py``)."""

    tool_returns: list[JudgedToolReturn] = []


AgentOutput = str
AgentCase = Case[AgentTurn, AgentOutput, None]

# The assistant's output tools, left out of the recorded calls (``record_tool_calls`` says why).
# Empty while the assistant's output is text; name the tool here when it gets an ``output_type``.
OUTPUT_TOOLS: frozenset[str] = frozenset()


async def run_assistant(turn: AgentTurn) -> AgentOutput:
    prompt = turn.prompt
    result = await get_agent(ASSISTANT).run(prompt, deps=make_deps())
    set_eval_attribute("tool_calls", record_tool_calls(result.all_messages(), OUTPUT_TOOLS))
    return str(result.output)


def record_tool_calls(messages: list[ModelMessage], output_tools: Collection[str] = ()) -> list[RecordedToolCall]:
    """The run's tool calls in the order the model made them, each paired with its answer.

    ``output_tools`` are left out: an agent with structured output delivers it through a tool
    whose arguments are the output, so a judge shown them as input would find every figure in
    the output "in the input". An answer that is not JSON is recorded as its string form:
    the recording must never fail the run it records.
    """
    answers: dict[str, tuple[Any, str]] = {}
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                answers[part.tool_call_id] = (to_jsonable_python(part.content, fallback=str), part.outcome)
    return [
        RecordedToolCall(name=part.tool_name, args=part.args_as_dict(), returned=answer, outcome=outcome)
        for message in messages
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name not in output_tools
        for answer, outcome in [answers.get(part.tool_call_id, (None, "no answer"))]
    ]


_PRINTED_RETURN_LIMIT = 2000


def describe_tool_call(call: RecordedToolCall) -> str:
    """One line for the report printout; a long answer (a search's full results) is cut short."""
    returned = to_json(call.returned).decode()
    if len(returned) > _PRINTED_RETURN_LIMIT:
        returned = f"{returned[:_PRINTED_RETURN_LIMIT]}... ({len(returned)} characters)"
    outcome = "" if call.outcome == "success" else f" {call.outcome}"
    return f"{call.name}({to_json(call.args).decode()}){outcome} -> {returned}"
