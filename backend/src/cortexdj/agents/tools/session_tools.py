import json

from pydantic_ai import RunContext

from cortexdj.agents.deps import AgentDeps
from cortexdj.services import session as session_service


async def list_sessions(ctx: RunContext[AgentDeps], limit: int = 100) -> str:
    """List recorded EEG sessions with friendly labels and dominant brain states.

    Returns a markdown list using stable display indices ("Session 01" ... "Session NN")
    and a human-readable label derived from each session's dominant emotional quadrant.
    The full session UUID is included on each line for use as an argument to other
    tools (e.g. `analyze_session`); never echo the UUID to the user.
    """
    summaries, total = await session_service.list_sessions_enriched(ctx.deps.db, limit=limit)

    if not summaries:
        return "No EEG sessions found in the database. Run `uv run seed-sessions` to populate."

    lines = [f"**{total} EEG sessions** (showing {len(summaries)}):\n"]
    for s in summaries:
        minutes = s["duration_seconds"] / 60
        lines.append(
            f"- **Session {s['display_index']:02d}** — {s['label']} · "
            f"{s['track_count']} tracks · ~{minutes:.0f} min  "
            f"<!-- id={s['id']} -->"
        )
    lines.append(
        "\n_Reference these by their **Session NN** label in your reply. "
        "Use the `id=...` value only as a tool argument._"
    )

    return "\n".join(lines)


async def analyze_session(ctx: RunContext[AgentDeps], session_id: str) -> str:
    """Get detailed brain state breakdown for a specific EEG session.

    Returns segment-by-segment arousal/valence scores, band powers,
    and associated track information.
    """
    detail = await session_service.get_session_detail(ctx.deps.db, session_id)

    if detail is None:
        return f"Session {session_id} not found."

    return json.dumps(detail, indent=2, default=str)
