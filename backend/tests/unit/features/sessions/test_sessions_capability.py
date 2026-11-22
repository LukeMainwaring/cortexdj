"""The sessions capability as the assistant sees it: which tools it offers, the reliability hedge, and the brain-context block."""

import pytest
from pydantic_ai.models.test import TestModel

from cortexdj.agents.registry import ASSISTANT, get_agent
from cortexdj.features.sessions.capability import SessionsCapability, render_brain_context
from cortexdj.features.sessions.eeg_model import override_eeg_model
from cortexdj.features.sessions.schema import BrainContext
from cortexdj.ml.checkpoint import AxisScores, QuadrantCheckpoint
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


def _static_instructions(checkpoint: QuadrantCheckpoint | None) -> str:
    with override_eeg_model(checkpoint):
        parts = SessionsCapability().get_instructions()
    return "\n".join(p for p in parts if isinstance(p, str))


def _checkpoint_scoring(mean: float) -> QuadrantCheckpoint:
    return make_quadrant_checkpoint(metrics=AxisScores(arousal=mean, valence=mean, mean=mean))


@pytest.mark.parametrize(
    ("score", "tier"),
    [(0.55, "a rough read"), (0.594, "a rough read"), (0.60, "a fair read"), (0.744, "a fair read")],
)
def test_the_hedge_is_worded_by_the_checkpoints_single_number(score: float, tier: str) -> None:
    static = _static_instructions(_checkpoint_scoring(score))

    assert f'"{tier}"' in static
    assert "one short clause" in static
    assert "never a paragraph" in static


@pytest.mark.parametrize(
    ("score", "printed", "tier"),
    [(0.5996, "0.60", '"a fair read"'), (0.7496, "0.75", "no reliability caveat")],
)
def test_the_tier_agrees_with_the_printed_score(score: float, printed: str, tier: str) -> None:
    static = _static_instructions(_checkpoint_scoring(score))

    assert f"is {printed} (chance is 0.50)" in static
    assert tier in static


@pytest.mark.parametrize("score", [0.75, 0.9])
def test_no_hedge_from_075_up(score: float) -> None:
    static = _static_instructions(_checkpoint_scoring(score))

    assert "a rough read" not in static
    assert "a fair read" not in static
    assert "no reliability caveat" in static


def test_with_no_checkpoint_the_most_cautious_tier_is_used() -> None:
    static = _static_instructions(None)

    assert '"a rough read"' in static
    assert "a fair read" not in static


def test_the_hedge_sits_in_the_static_instructions_and_brain_context_stays_the_only_callable() -> None:
    with override_eeg_model(_checkpoint_scoring(0.65)):
        parts = SessionsCapability().get_instructions()

    static = "\n".join(p for p in parts if isinstance(p, str))
    callables = [p for p in parts if not isinstance(p, str)]
    assert "a fair read" in static
    assert "0.65" in static
    assert len(callables) == 1


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
