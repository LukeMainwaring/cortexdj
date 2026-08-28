# Two-layer error handling for agent tools

**Status**: Accepted — 2026-07-16 (revised 2026-08-28: native `ToolFailed`)

An exception escaping a Pydantic AI tool body would abort the run and crash the
Vercel AI SDK SSE stream mid-response. Rather than wrap every tool in try/except
— which spreads error presentation across the tool layer and buries real bugs —
CortexDJ splits tool errors into two layers.

## Layer 1 — expected domain failures stay ordinary results

A failure the tool anticipates (Spotify not configured, token expired, no
sessions matched) is not an error; it is one of the outcomes the tool exists to
report. Those return a structured `{"error": ...}` dict from the tool body, and
the model reads it as a normal successful result and decides what to do next.
This layer is unchanged, and new tools should keep using it.

## Layer 2 — unexpected exceptions become native failed tool outcomes

Everything else propagates to `agents/hooks.py`. Its `tool_execute_error` hook
logs the traceback and raises `pydantic_ai.exceptions.ToolFailed`, which Pydantic
AI records as a native failed tool outcome: `ToolReturnPart.outcome == "failed"`,
the run continues, and the model explains the failure conversationally.

**Why `ToolFailed` and not a success payload containing `"error"`** — the
previous version of this decision returned a recovery *dict* from the hook, so an
unexpected crash was indistinguishable, in the message history and in traces,
from a layer-1 domain result. `ToolFailed` is the difference:

- The failure is typed in history (`outcome='failed'`) and shows as an error in
  Logfire, instead of hiding inside a successful-looking payload.
- It does **not** consume the tool's retry budget, unlike `ModelRetry`. A crashed
  tool is done, not worth re-attempting with tweaked arguments; bound repeated
  failures with run-level `UsageLimits` instead.
- The model still sees it and adapts, which is the property the recovery dict was
  bought for in the first place.

**Information disclosure** — the traceback goes to the application log; only the
tool name and the exception's *class name* reach the model. Exception messages
routinely carry connection strings, bearer tokens, and upstream response bodies,
and anything handed to the model is rendered to the user and stored in thread
history. This constraint is asserted in
`backend/tests/unit/agents/test_brain_agent_hooks.py`.

## The one sanctioned inline catch

`retrieval_tools.retrieve_tracks_from_brain_state` catches `DeapFileMissingError`
inline and returns it as a layer-1 payload. Layer 2 deliberately withholds the
exception's message — but this error's message *is* the actionable part (which
DEAP file is missing, and where to put it), and it contains no secrets. Catching
it at the tool lets that text reach the user verbatim. Adding a second inline
catch is a signal to re-examine this ADR, not to follow the precedent.
