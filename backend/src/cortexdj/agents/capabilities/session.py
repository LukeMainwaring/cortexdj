from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from cortexdj.agents.deps import AgentDeps
from cortexdj.agents.tools.session_tools import analyze_session, list_sessions

_TRAJECTORY_INSTRUCTIONS = """
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

When you call `list_sessions`, the UI renders the full session catalog
visually as a clickable panel directly beneath the tool call (cards with
labels, dominant state, quadrant distribution bars, duration, track
counts). **Do not enumerate the sessions in your text reply** — that
duplicates what the user already sees. Reply with at most one short
acknowledgement sentence and, if useful, 1–2 follow-up suggestions
(e.g. "click any card to analyze it, or tell me which session to dig
into"). Never repeat the per-session labels or counts in prose.

## Session Narrative

When summarizing an `analyze_session` result, narrate the listener's emotional
arc using the `trajectory_summary` fields, not just the single dominant state:

- **dwell_fractions** — percent of the session spent in each quadrant
  (`relaxed`, `calm`, `excited`, `stressed`).
- **transition_count** and **transitions** — how often and when the listener
  crossed quadrant boundaries; cite notable transitions with their timestamps.
- **path_length** — total distance travelled through affect space; a proxy for
  how dynamic the session was (low = steady, high = volatile).
- **dispersion** — spread around the centroid; a proxy for affective
  variability.
- **centroid** — the "average" point in (valence, arousal) space.

Prefer a two-to-three-sentence arc (e.g. "Started calm, shifted to excited
around 2:30, finished relaxed") over a flat list of statistics.
"""


@dataclass
class SessionCapability(AbstractCapability[AgentDeps]):
    def get_toolset(self) -> FunctionToolset[AgentDeps]:
        ts: FunctionToolset[AgentDeps] = FunctionToolset()
        ts.tool(list_sessions)
        ts.tool(analyze_session)
        return ts

    def get_instructions(self) -> str:
        return _TRAJECTORY_INSTRUCTIONS
