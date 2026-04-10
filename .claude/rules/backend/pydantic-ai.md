# Pydantic AI Rules

## Docs are split between two places

Pydantic AI's documentation lives in two places with **different content**:

1. **`docs/pydantic-ai-llms-full.txt`** — local pinned reference. API signatures,
   class docs, model provider APIs, the `pydantic_evals` surface. Refresh via
   the `updating-deps` skill (or `curl -sSL https://ai.pydantic.dev/llms-full.txt
   -o docs/pydantic-ai-llms-full.txt`).

2. **`https://ai.pydantic.dev/`** — conceptual guides, tutorials, example
   applications (chat, RAG, durable execution, graphs, A2A). Not cached locally;
   fetch ad-hoc with WebFetch.

**Rule of thumb:** if you're grepping `llms-full.txt` for a worked example or
tutorial and finding nothing, the content hasn't been deleted — it's on the web
docs site. Use WebFetch before giving up.

## Tool error handling

When adding a tool that makes external calls, prefer letting exceptions
propagate so `backend/src/cortexdj/agents/hooks.py::on_tool_execute_error`
catches them and returns a structured recovery payload. Don't wrap the whole
tool body in try/except unless you have a specific reason to handle a known
error shape differently (e.g., token-expired → reconnect prompt).
