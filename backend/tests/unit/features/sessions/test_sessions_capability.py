"""The sessions capability as the assistant sees it: which tools it offers, and the brain-context block."""

from pydantic_ai.models.test import TestModel

from cortexdj.agents.registry import ASSISTANT, get_agent
from cortexdj.features.sessions.capability import SessionsCapability, render_brain_context
from cortexdj.features.sessions.eeg_model import override_eeg_model
from cortexdj.features.sessions.schema import BrainContext
from cortexdj.ml.checkpoint import QuadrantCheckpoint
from tests.factories import make_deps, make_quadrant_checkpoint


async def _offered_tool_names(checkpoint: QuadrantCheckpoint | None) -> set[str]:
    model = TestModel(call_tools=[], custom_output_text="ok")
    with override_eeg_model(checkpoint), get_agent(ASSISTANT).override(model=model):
        await get_agent(ASSISTANT).run("hello", deps=make_deps())
    params = model.last_model_request_parameters
    assert params is not None, "TestModel has no recorded request parameters"
    return {t.name for t in params.function_tools}


async def test_offers_get_model_info_when_a_checkpoint_is_held() -> None:
    assert "get_model_info" in await _offered_tool_names(make_quadrant_checkpoint())


async def test_only_get_model_info_is_hidden_when_no_checkpoint_is_loaded() -> None:
    offered = await _offered_tool_names(None)

    assert "get_model_info" not in offered
    for always in ("list_sessions", "analyze_session", "explain_brain_state", "compare_sessions", "set_brain_context"):
        assert always in offered


def test_the_tool_list_describes_the_quadrant_classifier_not_the_old_ones() -> None:
    static = next(i for i in SessionsCapability().get_instructions() if isinstance(i, str))
    model_info_line = next(line for line in static.splitlines() if "**get_model_info**" in line)

    assert "quadrant classifier" in model_info_line
    assert "EEGNet" not in static


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
