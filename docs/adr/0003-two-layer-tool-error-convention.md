# Two-layer error handling for agent tools

**Status**: Accepted — 2026-07-16

An exception escaping a Pydantic AI tool body would crash the Vercel AI SDK SSE
stream mid-response. Rather than wrap every tool in try/except — which spreads
error presentation across the tool layer and buries real bugs — tools let
unanticipated exceptions propagate to `agents/hooks.py`, whose
`tool_execute_error` hook logs the traceback and returns a structured recovery
payload naming the tool and exception class, so the agent apologizes
conversationally instead of dying. Anticipated failures (Spotify not configured,
token expired) still return `{"error": ...}` dicts from the tool body itself.

The one sanctioned exception is `retrieval_tools.retrieve_tracks_from_brain_state`,
which catches `DeapFileMissingError` inline. The hook's payload deliberately
carries only the exception's class name, not its message — but this error's
message is the actionable part (which DEAP file is missing, and where to put
it). Catching it at the tool lets that text reach the user verbatim. Adding a
second inline catch is a signal to re-examine this ADR, not to follow the
precedent.
