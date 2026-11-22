"""The emotion quadrants: their fixed order, how a count of them picks one, and what each means.

The quadrants come from the arousal-valence model (Russell, 1980). Each explanation says what
the quadrant means for a listener and nothing about what their frequency bands did: no tool
measures band activity for a session, so do not add a band claim back.
"""

from collections.abc import Mapping

EMOTION_STATES: tuple[str, ...] = ("relaxed", "calm", "excited", "stressed")


def dominant_quadrant(counts: Mapping[str, float]) -> str:
    """The most counted quadrant, a tie going to the one first in ``EMOTION_STATES``.

    Every view that names a session's dominant quadrant (the card, the summary, the
    trajectory) picks it here, so a tie cannot name different quadrants in different places.
    A state outside ``EMOTION_STATES`` ranks after the known ones. Raises on empty ``counts``.
    """

    def rank(state: str) -> tuple[float, int]:
        order = EMOTION_STATES.index(state) if state in EMOTION_STATES else len(EMOTION_STATES)
        return (-counts[state], order)

    return min(counts, key=rank)


BRAIN_STATE_EXPLANATIONS: dict[str, str] = {
    "excited": (
        "Excited is high arousal with positive valence: an energized, engaged state. "
        "The music most likely felt lively and pleasant, the kind that pulls you in."
    ),
    "stressed": (
        "Stressed is high arousal with negative valence: a tense, overstimulated state. "
        "The music may have felt too intense or unsettling for your mood at the time."
    ),
    "relaxed": (
        "Relaxed is low arousal with positive valence: a calm, content state. "
        "The music most likely felt soothing and pleasant without demanding attention."
    ),
    "calm": (
        "Calm is low arousal with negative valence: a subdued, contemplative state. "
        "The music may have felt quiet or melancholy, more inward than uplifting."
    ),
}


def get_brain_state_explanation(state: str) -> str:
    return BRAIN_STATE_EXPLANATIONS.get(state, f"No detailed explanation available for state: {state}")
