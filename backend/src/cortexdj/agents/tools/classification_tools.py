from __future__ import annotations

import json

import torch
from pydantic_ai import RunContext

from cortexdj.agents.deps import AgentDeps
from cortexdj.core.paths import CHECKPOINTS_DIR


async def get_model_info(ctx: RunContext[AgentDeps]) -> str:
    """Get information about the loaded EEG classification model.

    Returns the model architecture, parameter count, and training metrics.
    """
    from cortexdj.ml.pretrained import PretrainedDualHead

    model = ctx.deps.eeg_model
    if model is None:
        return "No EEG model loaded. Run `uv run train-model` to train the model."

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
    from cortexdj.models.thread import Thread
    from cortexdj.schemas.agent_type import AgentType
    from cortexdj.schemas.thread import BrainContext

    if ctx.deps.thread_id is None:
        return "Cannot set brain context without a thread ID."

    updates = BrainContext(
        latest_session_id=session_id,
        dominant_mood=dominant_mood,
        avg_arousal=avg_arousal,
        avg_valence=avg_valence,
    )

    merged = await Thread.update_brain_context(
        ctx.deps.db,
        ctx.deps.thread_id,
        AgentType.CHAT.value,
        updates,
    )

    return f"Brain context updated: {merged.model_dump(exclude_none=True)}"
