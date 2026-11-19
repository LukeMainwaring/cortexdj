"""Unit tests for utils.emotion quadrant lookups (agent-facing copy)."""

from cortexdj.utils.emotion import (
    BRAIN_STATE_EXPLANATIONS,
    QUADRANT_DESCRIPTIONS,
    get_brain_state_explanation,
    quadrant_to_mood_description,
)


class TestQuadrantToMoodDescription:
    def test_known_quadrants(self) -> None:
        for state in ("excited", "stressed", "relaxed", "calm"):
            assert quadrant_to_mood_description(state) == QUADRANT_DESCRIPTIONS[state]

    def test_unknown_state_is_labeled_not_raised(self) -> None:
        assert quadrant_to_mood_description("bored") == "Unknown state: bored"


class TestGetBrainStateExplanation:
    def test_known_states_mention_a_frequency_band(self) -> None:
        for state, explanation in BRAIN_STATE_EXPLANATIONS.items():
            assert get_brain_state_explanation(state) == explanation
            assert any(band in explanation for band in ("alpha", "beta", "theta", "gamma"))

    def test_unknown_state_fallback(self) -> None:
        assert get_brain_state_explanation("bored").startswith("No detailed explanation")
