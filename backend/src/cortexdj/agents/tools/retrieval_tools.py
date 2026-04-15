import json

from pydantic_ai import RunContext

from cortexdj.agents.deps import AgentDeps
from cortexdj.services import retrieval as retrieval_service
from cortexdj.services.retrieval import DeapFileMissingError


async def retrieve_tracks_from_brain_state(
    ctx: RunContext[AgentDeps],
    session_id: str,
    k: int = 10,
) -> str:
    """Find new Spotify tracks whose audio matches a session's brain state in a learned joint space.

    Use this tool when the user asks to match music to how they were feeling,
    wants to discover new tracks grounded in a session's neural signature, or
    says things like "find songs that match this session" or "suggest music
    that sounds like my brain state". The returned tracks may be ones the user
    has never listened to before — unlike `find_relaxing_tracks` or
    `build_mood_playlist`, which only curate from the user's listening history
    filtered by a named quadrant.

    Args:
        session_id: The id of the EEG session to embed.
        k: Number of nearest tracks to return. Clamped to [1, 100] by the
            service layer; pass through unchanged.

    Returns a JSON payload with `session_id`, `k`, and a `tracks` list of
    `{spotify_id, title, artist, itunes_preview_url, similarity}` entries
    sorted by cosine similarity (higher is closer, max 1.0). When the
    retrieval index is empty the payload's `tracks` list is empty and a
    `note` field explains how to populate it. When the session's underlying
    DEAP data is missing on disk, the payload contains an `error` field
    that the agent should relay verbatim to the user.
    """
    try:
        hits = await retrieval_service.retrieve_similar_tracks(ctx.deps.db, session_id, k=k)
    except DeapFileMissingError as exc:
        # Server misconfig — the session row exists but its underlying DEAP
        # .dat file isn't on disk. The agent's default recovery template via
        # hooks.on_tool_execute_error only includes the exception class name,
        # which is useless to the user. Return a structured error so the
        # agent can relay an actionable message.
        return json.dumps(
            {
                "session_id": session_id,
                "error": "deap_data_missing",
                "message": (
                    f"Cannot embed session {session_id}: the underlying DEAP EEG file is not "
                    f"available on the server ({exc}). Ask the operator to place the DEAP .dat "
                    "files in backend/data/deap/ (see backend/data/DEAP_SETUP.md)."
                ),
            },
            indent=2,
        )

    if not hits:
        return json.dumps(
            {
                "session_id": session_id,
                "k": k,
                "tracks": [],
                "note": (
                    "The retrieval index is empty. To populate it, run "
                    "`uv run --directory backend seed-track-index` on the server. "
                    "Once populated, this tool returns tracks whose CLAP audio "
                    "embeddings are closest to the session's EEG embedding in "
                    "the learned contrastive space."
                ),
            },
            indent=2,
        )
    return json.dumps(
        {
            "session_id": session_id,
            "k": k,
            "tracks": retrieval_service.serialize_hits(hits),
        },
        indent=2,
    )
