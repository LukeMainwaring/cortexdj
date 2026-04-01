from __future__ import annotations

import json

from pydantic_ai import RunContext

from cortexdj.agents.deps import AgentDeps
from cortexdj.services import session as session_service


async def list_sessions(ctx: RunContext[AgentDeps], limit: int = 20) -> str:
    """List recorded EEG sessions with timestamps and emotion summaries.

    Shows available sessions from the database with participant info,
    recording dates, and dominant brain states.
    """
    sessions, total = await session_service.list_sessions(ctx.deps.db, limit=limit)

    if not sessions:
        return "No EEG sessions found in the database. Run `uv run seed-sessions` to populate."

    lines = [f"**{total} EEG sessions available** (showing {len(sessions)}):\n"]
    for s in sessions:
        lines.append(
            f"- **{s.id[:8]}...** | Participant {s.participant_id} | "
            f"{s.dataset_source} | {s.duration_seconds:.0f}s | "
            f"{s.recorded_at.strftime('%Y-%m-%d %H:%M')}"
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
