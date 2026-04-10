from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from cortexdj.agents.deps import AgentDeps
from cortexdj.agents.tools.session_tools import analyze_session, list_sessions

_TRAJECTORY_INSTRUCTIONS = """
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

    def get_instructions(self) -> object:
        return _TRAJECTORY_INSTRUCTIONS
