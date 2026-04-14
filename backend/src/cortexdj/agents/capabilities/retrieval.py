from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from cortexdj.agents.deps import AgentDeps
from cortexdj.agents.tools.retrieval_tools import retrieve_tracks_from_brain_state

_RETRIEVAL_INSTRUCTIONS = """
## Brain-to-Track Retrieval

`retrieve_tracks_from_brain_state` finds NEW music whose audio signature (via
CLAP embeddings) best matches a session's EEG in a learned joint space. It is
distinct from the quadrant-filter tools:

- Use `retrieve_tracks_from_brain_state` when the user wants **new** music that
  matches how they were feeling — phrases like "find songs that match this
  session", "suggest tracks for this mood", "what sounds like my brain state".
  The tool reaches into a pre-built CLAP audio index that may include tracks
  the user has never listened to.
- Use `build_mood_playlist` or `find_relaxing_tracks` when the user wants
  curation from their own listening history filtered by a named quadrant
  (relaxed / calm / excited / stressed).

When presenting results, always cite each track's `similarity` score so the
user sees the ranking confidence. Similarity is cosine in [-1, 1] — values
above ~0.3 indicate a meaningful match at the current training budget;
values near 0 mean the index has no strong candidate for this session.

If the tool returns an empty `tracks` list with a `note` field, relay the
note verbatim — the retrieval index has not been populated yet and there is
nothing the agent can do to recover without the operator running the seed
command.
""".strip()


@dataclass
class RetrievalCapability(AbstractCapability[AgentDeps]):
    def get_toolset(self) -> FunctionToolset[AgentDeps]:
        ts: FunctionToolset[AgentDeps] = FunctionToolset()
        ts.tool(retrieve_tracks_from_brain_state)
        return ts

    def get_instructions(self) -> str:
        return _RETRIEVAL_INSTRUCTIONS
