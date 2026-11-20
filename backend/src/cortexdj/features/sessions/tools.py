import json
from typing import Any

import torch
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry, ToolFailed

from cortexdj.agents.deps import AgentDeps
from cortexdj.core.paths import CHECKPOINTS_DIR
from cortexdj.features.sessions import service as session_service
from cortexdj.features.sessions.eeg_model import get_eeg_model
from cortexdj.features.sessions.emotion import get_brain_state_explanation
from cortexdj.features.sessions.schema import BrainContext
from cortexdj.ml.pretrained import PretrainedDualHead


def _unknown_session(session_id: str) -> ModelRetry:
    return ModelRetry(
        f"Session {session_id} not found. Pass a session UUID from the `id=...` value in `list_sessions` output."
    )


async def list_sessions(ctx: RunContext[AgentDeps], limit: int = 100) -> str:
    """List recorded EEG sessions with friendly labels and dominant brain states.

    Returns sessions newest-first by default — so `limit=1` is the right call
    for prompts like "show me my most recent EEG session". Display indices
    are stable ASC by recording time (Session 01 = oldest, Session NN = newest),
    so a low-numbered session is *not* a recent one. The full session UUID
    is included on each line for use as an argument to other tools
    (e.g. `analyze_session`); never echo the UUID to the user.

    The frontend renders these as a clickable card grid mirroring the same
    `limit`, so passing a small limit also visually narrows the panel.
    """
    summaries, total = await session_service.list_sessions_enriched(ctx.deps.db, limit=limit)

    if not summaries:
        return "No EEG sessions found in the database. Run `uv run --directory backend seed-sessions` to populate."

    lines = [f"**{total} EEG sessions** (showing {len(summaries)}):\n"]
    for s in summaries:
        minutes = s.duration_seconds / 60
        lines.append(
            f"- **Session {s.display_index:02d}** — {s.label} · "
            f"{s.track_count} tracks · ~{minutes:.0f} min  "
            f"<!-- id={s.id} -->"
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
        raise _unknown_session(session_id)

    return json.dumps(detail, indent=2, default=str)


async def explain_brain_state(ctx: RunContext[AgentDeps], session_id: str) -> str:
    """Explain what the brain was doing during a session in plain language.

    Provides educational context about EEG frequency bands and their
    relationship to emotional states.
    """
    if await session_service.get_session(ctx.deps.db, session_id) is None:
        raise _unknown_session(session_id)

    summary = await session_service.get_session_summary(ctx.deps.db, session_id)
    if summary is None:
        raise ToolFailed(f"No EEG segments found for session {session_id}.")

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
    for session_id in (session_id_1, session_id_2):
        if await session_service.get_session(ctx.deps.db, session_id) is None:
            raise _unknown_session(session_id)

    result = await session_service.compare_sessions(ctx.deps.db, session_id_1, session_id_2)

    if result is None:
        raise ToolFailed("One or both sessions have no segments.")

    return json.dumps(result, indent=2, default=str)


async def find_relaxing_tracks(ctx: RunContext[AgentDeps], limit: int = 20) -> dict[str, Any]:
    """Find tracks that triggered calm or relaxed brain states.

    Queries historical EEG data to find tracks associated with
    high valence and low arousal (the relaxation quadrant).
    """
    relaxed = await session_service.find_tracks_by_mood(ctx.deps.db, "relaxed", limit)
    calm = await session_service.find_tracks_by_mood(ctx.deps.db, "calm", limit)

    all_tracks = relaxed + calm
    if not all_tracks:
        return {"tracks": [], "count": 0, "message": "No tracks with relaxation data found."}

    seen: set[str] = set()
    unique = []
    for t in all_tracks:
        if t.track_id not in seen:
            seen.add(t.track_id)
            unique.append(t)

    return {
        "tracks": [
            {
                "title": t.title,
                "artist": t.artist,
                "dominant_state": t.dominant_state,
                "avg_arousal": round(t.avg_arousal, 2),
                "avg_valence": round(t.avg_valence, 2),
            }
            for t in unique[:limit]
        ],
        "count": len(unique[:limit]),
        "total_available": len(unique),
    }


async def get_model_info(ctx: RunContext[AgentDeps]) -> str:
    """Get information about the loaded EEG classification model.

    Returns the model architecture, parameter count, and training metrics.
    """
    model = get_eeg_model()
    if model is None:
        raise ToolFailed(
            "No EEG model loaded. Run `./backend/scripts/download-checkpoints.sh` first, then restart the backend."
        )

    is_pretrained = isinstance(model, PretrainedDualHead)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    checkpoint_name = "cbramod_best.pt" if is_pretrained else "eegnet_best.pt"
    checkpoint_path = CHECKPOINTS_DIR / checkpoint_name
    metrics = {}
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        metrics = checkpoint.get("metrics", {})

    if is_pretrained:
        architecture = "CBraMod Pretrained + Dual-Head Fine-Tuned"
        input_desc = "Raw EEG (32 channels x 800 samples at 200Hz)"
    else:
        architecture = "EEGNet Dual-Head Classifier"
        input_desc = "Differential entropy features (32 channels x 5 bands = 160 features)"

    info = {
        "architecture": architecture,
        "input": input_desc,
        "outputs": {
            "arousal_head": "Binary classification (low/high arousal)",
            "valence_head": "Binary classification (low/high valence)",
        },
        "parameters": {
            "total": param_count,
            "trainable": trainable_count,
        },
        "training_metrics": metrics or "No metrics available (model not yet trained)",
    }

    return json.dumps(info, indent=2)


async def set_brain_context(
    ctx: RunContext[AgentDeps],
    session_id: str | None = None,
    dominant_mood: str | None = None,
    avg_arousal: float | None = None,
    avg_valence: float | None = None,
) -> str:
    """Set or update the brain state context for this conversation.

    Call when the user references a specific session or mood. Context
    persists across messages and influences playlist recommendations.
    Only provide fields that are being set or changed.
    """
    if ctx.conversation_id is None:
        raise ToolFailed("Cannot set brain context without a thread ID.")

    updates = BrainContext(
        latest_session_id=session_id,
        dominant_mood=dominant_mood,
        avg_arousal=avg_arousal,
        avg_valence=avg_valence,
    )

    merged = await session_service.merge_brain_context(ctx.deps.db, ctx.conversation_id, updates)

    return f"Brain context updated: {merged.model_dump(exclude_none=True)}"
