# Emotion quadrants based on arousal-valence model (Russell, 1980)
BRAIN_STATE_EXPLANATIONS: dict[str, str] = {
    "excited": (
        "Your brain showed high beta activity (14-30 Hz) and elevated alpha power, "
        "indicating an alert, focused state with positive engagement. "
        "This pattern is typical during upbeat, energizing music."
    ),
    "stressed": (
        "Your brain showed elevated beta and gamma activity with suppressed alpha waves. "
        "This suggests mental tension or overstimulation — the music may have been "
        "too intense or dissonant for your current state."
    ),
    "relaxed": (
        "Your brain showed strong alpha activity (8-14 Hz) with reduced beta power, "
        "the classic signature of relaxation. Alpha waves are associated with "
        "a calm, meditative state — this music was genuinely soothing."
    ),
    "calm": (
        "Your brain showed increased theta activity (4-8 Hz) with moderate alpha power. "
        "This suggests a dreamy, introspective state — not quite relaxed, "
        "more contemplative or slightly detached from the stimulus."
    ),
}


def get_brain_state_explanation(state: str) -> str:
    return BRAIN_STATE_EXPLANATIONS.get(state, f"No detailed explanation available for state: {state}")
