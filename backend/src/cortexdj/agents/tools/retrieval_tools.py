from __future__ import annotations

import json

from pydantic_ai import RunContext

from cortexdj.agents.deps import AgentDeps
from cortexdj.services import retrieval as retrieval_service


async def retrieve_tracks_from_brain_state(
    ctx: RunContext[AgentDeps],
    session_id: str,
    k: int = 10,
) -> str:
    """Find new Spotify tracks whose audio matches a session's brain state.

    Runs the session's raw EEG through the contrastive EegCLAPEncoder to
    produce a 512-d query vector in the joint EEG↔audio embedding space,
    then returns the top-k nearest tracks from the precomputed CLAP audio
    index (pgvector HNSW cosine search). The returned tracks may be ones
    the user has never listened to before — unlike history-filter tools
    like `find_relaxing_tracks` or `build_mood_playlist`.

    Use when the user asks to match music to how they were feeling,
    suggests finding new music by brain state, or wants recommendations
    grounded in a session's neural signature rather than a specific
    arousal/valence label.

    Args:
        session_id: The id of the EEG session to embed.
        k: Number of nearest tracks to return. Clamped to [1, 100].

    Returns a JSON payload with `session_id`, `k`, and a `tracks` list
    of `{spotify_id, title, artist, itunes_preview_url, similarity}`
    entries sorted by cosine similarity (higher is closer, max 1.0).
    When the retrieval index is empty the payload's `tracks` list is
    empty and a `note` field explains how to populate it.
    """
    hits = await retrieval_service.retrieve_similar_tracks(ctx.deps.db, session_id, k=k)
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
