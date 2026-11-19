---
paths:
  - "backend/src/cortexdj/agents/**/*.py"
  - "backend/src/cortexdj/features/**/tools.py"
  - "backend/src/cortexdj/features/**/capability.py"
  - "backend/src/cortexdj/features/chat/stream.py"
  - "backend/src/cortexdj/features/chat/serialization.py"
  - "backend/tests/evals/**/*.py"
---

# Pydantic AI rules

## Three doc sources, in this order

1. **Skill `building-pydantic-ai-agents`**: load it first.
2. **Installed source**: `backend/.venv/lib/python3.*/site-packages/pydantic_ai/` and
   `pydantic_evals/`. Grep it for exact API signatures, class fields, and defaults the skill
   leaves out; its docstrings are what the online API reference is generated from, and it
   always matches the installed version.
3. **`https://pydantic.dev/docs/ai/`**: WebFetch for guides, tutorials, and example apps
   when the first two don't answer.

The `updating-deps` skill repairs the skill symlink after a Python bump.

## Instructions: static rules, dynamic context

`build_agent` adds the `Caching` capability to every agent, and the provider's cache
breakpoint falls after the last *static* instruction. A capability's `get_instructions()`
returns its rules as a literal string and returns a callable only for what varies per run
(`[RULES, per_run_context]`), sync unless it does I/O; pydantic-ai sorts static parts before
dynamic ones. Rules folded into a callable are never in the cross-run cache prefix. Rationale
and trade-offs: `docs/adr/0002-prompt-caching-in-the-shared-recipe.md`.

## Structured output

An agent that must sometimes answer with an object (a plan, a report) takes it as an output
tool, with `str` kept in the output union so ordinary turns still end as text. The chat UI
renders a tool call as a collapsible row, so a tool panel renders the object from the call's
`input`; ask the model for the object alone. Never raise `ModelRetry` to demand text alongside
the output tool: a real model resubmits the bare tool call, the output retry budget (default 1)
runs out, and the run ends in `UnexpectedModelBehavior` after the object already streamed. Name
output tools in `OUTPUT_TOOLS` (`tests/evals/agent_run.py`) so the evals record the object the
model produced.

An `after_output_process` hook is outside the output error hook (`on_output_process_error` in
`agents/hooks.py`): an exception there aborts the run and cuts the SSE stream. Work on the
validated object, persisting it say, goes in an output validator that skips
`ctx.partial_output`.
