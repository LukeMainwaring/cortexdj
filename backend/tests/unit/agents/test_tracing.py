"""An agent run is traced under the Pydantic AI instrumentor the app attaches (no token needed)."""

from logfire.testing import CaptureLogfire
from pydantic_ai.models.test import TestModel

import cortexdj.app  # noqa: F401  # attaches the instrumentors, as a real process would
from cortexdj.agents.registry import ASSISTANT, get_agent
from tests.factories import make_deps


async def test_agent_run_is_traced(capfire: CaptureLogfire) -> None:
    agent = get_agent(ASSISTANT)
    with agent.override(model=TestModel(call_tools=[])):
        await agent.run("hello", deps=make_deps())

    spans = capfire.exporter.exported_spans_as_dict()
    assert any(span["attributes"].get("gen_ai.agent.name") == ASSISTANT for span in spans)
