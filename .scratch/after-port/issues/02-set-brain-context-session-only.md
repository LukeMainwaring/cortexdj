# 02: `set_brain_context` with only a session id

**Status:** needs-planning
**Planning approach:** grill-with-docs

**Planning gate:** Refine against the current implementation with the user before
implementation. Update scope, acceptance criteria, and dependencies; record
settled domain terms and architectural decisions where needed. Return to
ready-for-agent once the user agrees the plan is settled.

**What is wrong:** two things in `set_brain_context` (`features/sessions/tools.py`).

- It stores whatever session id it is given. A model that passed an id of no session got
  "Brain context updated" back, and the thread's context pointed at nothing until the model
  noticed and called again.
- The header's brain-context badge (`frontend/features/sessions/brain-context-badge.tsx`)
  draws only the mood, arousal, and valence fields. The model usually passes the session id
  alone, so the context is stored and the badge shows nothing. Asking the assistant to set the
  mood and averages makes it appear.

**A likely shape:** when the call gives a session id, the tool looks the session up, answers
`ModelRetry` for an unknown id as the other session tools do, and fills the mood and averages
from the session's summary unless the call gave its own.
