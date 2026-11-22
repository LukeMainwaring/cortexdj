"""EEG sessions for the assistant: listing, analysis, comparison, the model, the reliability hedge, and brain context.

The reliability hedge is worded from the checkpoint loaded at startup. pydantic-ai reads a
capability's instructions once, when the registry builds the agent, which is after the
startup hooks ran, so the hedge is fixed for the life of the process and stays static text.

The conversation's brain context is read from the database before every model request,
not carried on ``AgentDeps``: ``set_brain_context`` may change it mid-run, and the next
request of the same run must see the change. It is the only dynamic part of these
instructions; every rule stays in the static text so it sits in the cached prefix
(``docs/adr/0002-prompt-caching-in-the-shared-recipe.md``).
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic_ai import RunContext, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from cortexdj.agents.deps import AgentDeps
from cortexdj.features.sessions import service as session_service
from cortexdj.features.sessions.eeg_model import get_eeg_model
from cortexdj.features.sessions.schema import BrainContext
from cortexdj.features.sessions.tools import (
    analyze_session,
    compare_sessions,
    explain_brain_state,
    find_relaxing_tracks,
    get_model_info,
    list_sessions,
    set_brain_context,
)
from cortexdj.ml.checkpoint import QuadrantCheckpoint

_SESSIONS_INSTRUCTIONS = """
## EEG Analysis

- **list_sessions** — Show recorded EEG sessions with timestamps and emotion summaries
- **analyze_session** — Detailed breakdown of a session's brain states (per-segment timeline, band powers, tracks)
- **explain_brain_state** — Plain-language explanation of what the brain was doing during a session
- **compare_sessions** — Compare brain patterns across two sessions
- **get_model_info** — Describe the quadrant classifier: its features, split, accuracy and training recipe
- **set_brain_context** — Set the active brain state context for this conversation
- **find_relaxing_tracks** — Find tracks that triggered calm/relaxed brain states from EEG data

- Include session/segment IDs in responses for reference
- Proactively call set_brain_context when the user references a session or mood

## Session References

When the user asks about EEG sessions:

- Always refer to a session by its **Session NN** label (e.g. "Session 07"),
  never by its UUID. UUIDs are tool arguments only — they must not appear in
  user-facing prose, even in code spans or footnotes.
- Never mention the participant ID, dataset source ("deap"), or recorded-at
  timestamp. These are internal seeding details, not user-facing facts.
- When the user says something like "analyze session 7" or "show me session
  03", look up the matching UUID from the most recent `list_sessions` output
  (the `id=...` HTML comment on each line) and pass that UUID to
  `analyze_session`. The user only ever sees the label.

## Listing Sessions

When you call `list_sessions`, the UI renders the result visually as a
clickable card panel directly beneath the tool call (cards with labels,
dominant state, quadrant distribution bars, duration, track counts). The
panel **mirrors the `limit` you passed** — call `list_sessions(limit=1)`
for "show me my most recent session", `limit=5` for "show me my last
five", and only use the larger default when the user asks for the full
catalog. Sessions are returned newest-first by default.

**Do not enumerate the sessions in your text reply** — that duplicates
what the user already sees. Reply with at most one short acknowledgement
sentence and, if useful, 1–2 follow-up suggestions (e.g. "click any card
to analyze it, or tell me which session to dig into"). Never repeat the
per-session labels or counts in prose.

## Session Narrative

When summarizing an `analyze_session` result, narrate the listener's emotional
arc track by track using the `trajectory_summary` fields, not just the single
dominant state. Every field is counted over the session's tracks, one point per
track at the mean of its segments:

- **dwell_fractions** — share of the session's tracks in each quadrant
  (`relaxed`, `calm`, `excited`, `stressed`).
- **transition_count** and **transitions** — how often the listener moved to a
  different quadrant from one track to the next; a transition's `time` is the
  start of the track it lands on. Cite notable transitions by track (its
  position and title from `tracks`), not by timestamp.
- **path_length** — total distance travelled through affect space from track
  to track; a proxy for how dynamic the session was (low = steady, high =
  volatile).
- **dispersion** — spread of the tracks around the centroid; a proxy for
  affective variability.
- **centroid** — the "average" point in (arousal, valence) space.

Prefer a two-to-three-sentence arc (e.g. "Started calm, shifted to excited
after the third track, finished relaxed") over a flat list of statistics.
""".strip()

# The single number's cut-offs: below the first "a rough read", up to the second "a fair read",
# from the second up no hedge.
_FAIR_FROM = 0.60
_NO_HEDGE_FROM = 0.75


def render_reliability(checkpoint: QuadrantCheckpoint | None) -> str:
    """The "Reliability" block: how to hedge a session's quadrants, worded by the checkpoint's single number.

    The single number is the checkpoint's balanced accuracy, the mean of arousal and valence,
    averaged over the participants it holds. With no checkpoint loaded the most cautious tier
    is used, since nothing vouches for the stored quadrants.
    """
    if checkpoint is None:
        basis = "No quadrant-classifier checkpoint is loaded, so nothing vouches for the stored quadrants."
        tier: str | None = "a rough read"
    else:
        score = checkpoint.summary.mean.mean
        basis = (
            f"The quadrant classifier's balanced accuracy, averaged over arousal and valence and over "
            f"its participants, is {score:.2f} (chance is 0.50)."
        )
        tier = "a rough read" if score < _FAIR_FROM else "a fair read" if score < _NO_HEDGE_FROM else None
    if tier is None:
        rule = "That is reliable enough to state the quadrants plainly: add no reliability caveat."
    else:
        rule = (
            f'When you analyze or explain a session, say once, in one short clause, that its quadrants are "{tier}" '
            f'(for example: "this is {tier}, but the session leans calm"). One clause per analysis, never a '
            "paragraph, and do not repeat it for each track."
        )
    return f"## Reliability\n\n{basis} {rule}"


# Hidden while no checkpoint is loaded, so the model never offers a tool that can only fail.
_MODEL_TOOLS = frozenset({get_model_info.__name__})


def render_brain_context(context: BrainContext | None) -> str:
    """The "Active Brain Context" block, or an empty string when nothing is set."""
    if context is None:
        return ""
    parts = []
    if context.latest_session_id:
        parts.append(f"Active session: {context.latest_session_id}")
    if context.dominant_mood:
        parts.append(f"Dominant mood: {context.dominant_mood}")
    if context.avg_arousal is not None:
        parts.append(f"Avg arousal: {context.avg_arousal:.2f}")
    if context.avg_valence is not None:
        parts.append(f"Avg valence: {context.avg_valence:.2f}")
    if not parts:
        return ""
    return (
        "\n\n## Active Brain Context\n"
        + "\n".join(f"- {p}" for p in parts)
        + "\n\nUse this context to personalize analysis and playlist recommendations."
    )


async def _brain_context_for_request(ctx: RunContext[AgentDeps]) -> str:
    if ctx.conversation_id is None:
        return ""
    return render_brain_context(await session_service.get_brain_context(ctx.deps.db, ctx.conversation_id))


@dataclass
class SessionsCapability(AbstractCapability[AgentDeps]):
    # Tools whose large list results the history processor may compact in prior
    # turns (see agents/history_processor.py).
    summarizable_tools: frozenset[str] = frozenset({"find_relaxing_tracks"})

    def get_toolset(self) -> FunctionToolset[AgentDeps]:
        # The tools share one AgentDeps.db session, and an AsyncSession cannot run
        # overlapping operations; pydantic-ai would otherwise run a response's several
        # tool calls concurrently.
        ts: FunctionToolset[AgentDeps] = FunctionToolset(sequential=True)
        ts.tool(list_sessions)
        ts.tool(analyze_session)
        ts.tool(explain_brain_state)
        ts.tool(compare_sessions)
        ts.tool(find_relaxing_tracks)
        ts.tool(get_model_info)
        ts.tool(set_brain_context)
        return ts

    def get_instructions(self) -> list[str | Callable[[RunContext[AgentDeps]], Awaitable[str]]]:
        # Both strings are static and sit in the cached prefix; brain context is the only
        # per-request part (ADR 0002).
        return [_SESSIONS_INSTRUCTIONS, render_reliability(get_eeg_model()), _brain_context_for_request]

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDeps],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Hide the model tools when no quadrant-classifier checkpoint is loaded."""
        if get_eeg_model() is None:
            return [td for td in tool_defs if td.name not in _MODEL_TOOLS]
        return tool_defs
