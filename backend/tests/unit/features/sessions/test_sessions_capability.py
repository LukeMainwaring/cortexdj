"""The sessions capability as the assistant sees it: which tools it offers, and the brain-context block."""

from pydantic_ai.models.test import TestModel

from cortexdj.agents.registry import ASSISTANT, get_agent
from cortexdj.features.sessions.capability import render_brain_context
from cortexdj.features.sessions.eeg_model import override_eeg_model
from cortexdj.features.sessions.schema import BrainContext
from cortexdj.ml.model import EEGNetClassifier
from cortexdj.ml.predict import EEGModel
from tests.factories import make_deps


async def _offered_tool_names(eeg_model: EEGModel | None) -> set[str]:
    model = TestModel(call_tools=[], custom_output_text="ok")
    with override_eeg_model(eeg_model), get_agent(ASSISTANT).override(model=model):
        await get_agent(ASSISTANT).run("hello", deps=make_deps())
    params = model.last_model_request_parameters
    assert params is not None, "TestModel has no recorded request parameters"
    return {t.name for t in params.function_tools}


async def test_hides_get_model_info_when_no_eeg_model_is_loaded() -> None:
    assert "get_model_info" not in await _offered_tool_names(None)


async def test_offers_get_model_info_when_an_eeg_model_is_held() -> None:
    assert "get_model_info" in await _offered_tool_names(EEGNetClassifier())


async def test_set_brain_context_is_offered_without_a_model() -> None:
    assert "set_brain_context" in await _offered_tool_names(None)


async def test_session_and_insight_tools_are_offered_without_a_model() -> None:
    offered = await _offered_tool_names(None)

    for always in ("list_sessions", "analyze_session", "explain_brain_state", "compare_sessions"):
        assert always in offered


def test_no_context_renders_nothing() -> None:
    assert render_brain_context(None) == ""
    assert render_brain_context(BrainContext()) == ""


def test_renders_each_set_field_as_a_line() -> None:
    block = render_brain_context(
        BrainContext(latest_session_id="sess-1", dominant_mood="relaxed", avg_arousal=0.25, avg_valence=0.8)
    )

    assert "## Active Brain Context" in block
    assert "- Active session: sess-1" in block
    assert "- Dominant mood: relaxed" in block
    assert "- Avg arousal: 0.25" in block
    assert "- Avg valence: 0.80" in block


def test_a_zero_score_is_rendered_not_skipped() -> None:
    block = render_brain_context(BrainContext(avg_arousal=0.0))

    assert "- Avg arousal: 0.00" in block
    assert "Active session" not in block
