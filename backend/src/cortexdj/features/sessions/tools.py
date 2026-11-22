import json
from dataclasses import asdict
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry, ToolFailed

from cortexdj.agents.deps import AgentDeps
from cortexdj.features.sessions import service as session_service
from cortexdj.features.sessions.eeg_model import get_eeg_model
from cortexdj.features.sessions.emotion import get_brain_state_explanation
from cortexdj.features.sessions.schema import BrainContext
from cortexdj.ml.dataset import NUM_EEG_CHANNELS


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
    """Describe the quadrant classifier that scored the sessions: its features, split, accuracy and training recipe.

    Call when the user asks how the brain-state readings are made or how reliable they are.
    """
    checkpoint = get_eeg_model()
    if checkpoint is None:
        raise ToolFailed(
            "No quadrant-classifier checkpoint loaded. Run `uv run --directory backend train-classifier`, "
            "then restart the backend."
        )

    recipe = checkpoint.recipe
    summary = checkpoint.summary
    recipe_values = asdict(recipe)
    recipe_values["bands"] = {band.name: [band.low, band.high] for band in recipe.bands}
    info = {
        "model": (
            f"One linear layer on {NUM_EEG_CHANNELS * len(recipe.bands)} band features, with the training "
            "fold's feature scaling built in; two outputs, the logits of a high arousal and a high valence "
            "rating. A segment's score is the sigmoid of its logit, and its quadrant is cut at 0.5 on each axis."
        ),
        "features": {
            "count": NUM_EEG_CHANNELS * len(recipe.bands),
            "description": (
                f"Differential entropy (half the log of the band-passed variance) of each of "
                f"{NUM_EEG_CHANNELS} channels in each band, over one {recipe.window_seconds}-second window"
            ),
            "bands_hz": recipe_values["bands"],
        },
        "models": (
            f"One set of {recipe.n_folds} fold models per participant, trained only on that participant's "
            "own trials; no model reads a participant it was not trained on"
        ),
        "participants": len(checkpoint.participants),
        "split": (
            f"Within-subject: a stratified {recipe.n_folds}-fold over each participant's trials, every window "
            "of a trial on one side, every trial held out once. Each stored segment was scored by the fold "
            "model that never saw its trial."
        ),
        "balanced_accuracy": {
            "arousal": asdict(summary.arousal),
            "valence": asdict(summary.valence),
            "mean": asdict(summary.mean),
            "chance": 0.5,
            "scored_over": "each participant's held-out windows, then averaged over participants (std across them)",
        },
        "training_recipe": recipe_values,
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

    # Only the arguments the model gave: a field passed to BrainContext counts as set even
    # when it is None, and the merge would then clear what an earlier call stored.
    given = {
        "latest_session_id": session_id,
        "dominant_mood": dominant_mood,
        "avg_arousal": avg_arousal,
        "avg_valence": avg_valence,
    }
    updates = BrainContext.model_validate({name: value for name, value in given.items() if value is not None})

    merged = await session_service.merge_brain_context(ctx.deps.db, ctx.conversation_id, updates)

    return f"Brain context updated: {merged.model_dump(exclude_none=True)}"
