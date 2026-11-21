"""The retrieval capability as the assistant sees it: its one tool, and static instructions that name it."""

from pydantic_ai.models.test import TestModel

from cortexdj.agents.registry import ASSISTANT, get_agent
from cortexdj.features.retrieval.capability import RetrievalCapability
from tests.factories import make_deps


def test_the_toolset_offers_only_the_retrieval_tool() -> None:
    toolset = RetrievalCapability().get_toolset()

    assert set(toolset.tools) == {"retrieve_tracks_from_brain_state"}


async def test_the_assistant_offers_the_retrieval_tool() -> None:
    model = TestModel(call_tools=[], custom_output_text="ok")
    with get_agent(ASSISTANT).override(model=model):
        await get_agent(ASSISTANT).run("hello", deps=make_deps())
    params = model.last_model_request_parameters
    assert params is not None, "TestModel has no recorded request parameters"

    assert "retrieve_tracks_from_brain_state" in {t.name for t in params.function_tools}


def test_the_instructions_are_a_static_string_naming_the_tool() -> None:
    instructions = RetrievalCapability().get_instructions()

    assert isinstance(instructions, str)
    assert "retrieve_tracks_from_brain_state" in instructions
    assert RetrievalCapability().get_instructions() == instructions
