from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from cortexdj.agents.deps import AgentDeps
from cortexdj.features.notes.tools import create_note, list_notes

_NOTES_INSTRUCTIONS = """
## Notes

You can manage the user's notes. Call `list_notes` to retrieve their existing
notes (returned newest-first), and `create_note` with a title and content to
save a new one. Use these tools whenever the user wants to review, add, or
remember something.
""".strip()


@dataclass
class NotesCapability(AbstractCapability[AgentDeps]):
    # Tools whose large list results the history processor may compact in prior
    # turns (see agents/history_processor.py). Declared here so feature knowledge
    # stays in the slice, not hardcoded in the shared runtime.
    summarizable_tools: frozenset[str] = frozenset({"list_notes"})

    def get_toolset(self) -> FunctionToolset[AgentDeps]:
        # The tools share one AgentDeps.db session, and an AsyncSession cannot run
        # overlapping operations; pydantic-ai would otherwise run a response's several
        # tool calls concurrently.
        ts: FunctionToolset[AgentDeps] = FunctionToolset(sequential=True)
        ts.tool(list_notes)
        ts.tool(create_note)
        return ts

    def get_instructions(self) -> str:
        return _NOTES_INSTRUCTIONS
