"""What each emotion quadrant means, for ``explain_brain_state``.

The quadrants come from the arousal-valence model (Russell, 1980). Each explanation says what
the quadrant means for a listener and nothing about what their frequency bands did: no tool
measures band activity for a session, so do not add a band claim back.
"""

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
