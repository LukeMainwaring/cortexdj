"""The canned quadrant explanations: what each quadrant means, with no claim about what the bands did."""

import pytest

from cortexdj.features.sessions.emotion import get_brain_state_explanation

_BAND_WORDS = ("delta", "theta", "alpha", "beta", "gamma", "hz", "waves", "power")


@pytest.mark.parametrize(
    ("state", "meaning"),
    [("relaxed", "low arousal"), ("calm", "low arousal"), ("excited", "high arousal"), ("stressed", "high arousal")],
)
def test_each_quadrant_is_explained_by_what_it_means(state: str, meaning: str) -> None:
    assert meaning in get_brain_state_explanation(state).lower()


@pytest.mark.parametrize("state", ["relaxed", "calm", "excited", "stressed"])
def test_no_explanation_claims_band_activity(state: str) -> None:
    explanation = get_brain_state_explanation(state).lower()

    assert not [word for word in _BAND_WORDS if word in explanation]
