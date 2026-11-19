"""A chat turn is one connected trace: the request span is an ancestor of the agent run.

Goes through the real route with a model double, so the spans come from the FastAPI and
Pydantic AI instrumentors the app attached, not from anything the test set up.
"""

from typing import Any

from httpx import AsyncClient
from logfire.testing import CaptureLogfire
from pydantic_ai.models.function import FunctionModel

from cortexdj.agents.registry import ASSISTANT, get_agent
from tests.factories import make_chat_request
from tests.integration.features.chat.conftest import CHAT_URL, reply_ok


def _ancestors(span: dict[str, Any], by_span_id: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    while span["parent"]:
        span = by_span_id[span["parent"]["span_id"]]
        chain.append(span)
    return chain


async def test_chat_turn_is_one_connected_trace(client: AsyncClient, capfire: CaptureLogfire) -> None:
    with get_agent(ASSISTANT).override(model=FunctionModel(stream_function=reply_ok)):
        response = await client.post(CHAT_URL, json=make_chat_request("t-trace", "hi"))
    assert response.status_code == 200

    spans = capfire.exporter.exported_spans_as_dict()
    by_span_id = {span["context"]["span_id"]: span for span in spans}
    [request] = [span for span in spans if span["name"] == f"POST {CHAT_URL}"]
    [agent_run] = [span for span in spans if span["name"] == f"invoke_agent {ASSISTANT}"]

    # Same trace, and the request span is up the agent run's parent chain: what lets a
    # developer click from the HTTP request to the model calls it caused.
    assert agent_run["context"]["trace_id"] == request["context"]["trace_id"]
    assert request in _ancestors(agent_run, by_span_id)
    # The run span already names the thread; nothing else needs a span of its own (ADR 0003).
    assert agent_run["attributes"]["gen_ai.conversation.id"] == "t-trace"
