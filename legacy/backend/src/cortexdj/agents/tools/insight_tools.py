import json

from pydantic_ai import RunContext

from cortexdj.agents.deps import AgentDeps
from cortexdj.models.eeg_segment import EegSegment
from cortexdj.models.session import Session
from cortexdj.services import eeg_processing as eeg_service
from cortexdj.utils.emotion import get_brain_state_explanation


async def explain_brain_state(ctx: RunContext[AgentDeps], session_id: str) -> str:
    """Explain what the brain was doing during a session in plain language.

    Provides educational context about EEG frequency bands and their
    relationship to emotional states.
    """
    session = await Session.get(ctx.deps.db, session_id)
    if session is None:
        return f"Session {session_id} not found."

    summary = await EegSegment.get_session_summary(ctx.deps.db, session_id)
    if "error" in summary:
        return f"No EEG segments found for session {session_id}."

    dominant = summary["dominant_state"]
    explanation = get_brain_state_explanation(dominant)

    lines = [
        f"**Brain State Analysis for Session {session_id[:8]}...**\n",
        f"**Dominant State:** {dominant.capitalize()}",
        f"**Average Arousal:** {summary['avg_arousal']:.2f} (0 = very low, 1 = very high)",
        f"**Average Valence:** {summary['avg_valence']:.2f} (0 = negative, 1 = positive)",
        f"**State Distribution:** {json.dumps(summary['state_distribution'])}",
        f"\n**What was happening in your brain:**\n{explanation}",
    ]

    return "\n".join(lines)


async def compare_sessions(ctx: RunContext[AgentDeps], session_id_1: str, session_id_2: str) -> str:
    """Compare brain state patterns across two EEG sessions.

    Highlights differences in arousal, valence, and dominant states
    to identify how the listener's response changed.
    """
    result = await eeg_service.compare_sessions(ctx.deps.db, session_id_1, session_id_2)

    if "error" in result:
        return str(result["error"])

    return json.dumps(result, indent=2, default=str)
