# 0002: Prompt caching in the shared recipe, with the rules static and the per-run context dynamic

**Status:** accepted (2026-10-17)

## Context

Every chat turn resends the tool definitions, the system prompt, each capability's instructions,
and the whole thread history, and a turn with tool calls resends them once per round-trip.
Providers that cache a prompt prefix bill a cache read at a fraction of the input price (Anthropic:
a tenth, with a cache write at 1.25 times). Pydantic AI's `Caching` capability sets one portable
`cache` model setting and places the provider's breakpoints itself: after the tools, after the last
*static* instruction, and, when `messages=True`, at the end of the conversation.

The breakpoint after the static instructions is the one that matters across runs, and it only
covers text Pydantic AI can tell is static: a literal string, not the output of a callable. A
capability that returns its whole prompt, rules and per-run context alike, from one callable keeps
its rules out of the cross-run prefix.

## Decision

- **`build_agent` adds `Caching(messages=cache_messages)` to every agent.** One place, like the
  other recipe pieces. `cache_messages` defaults to `True`; an agent that makes one request per run
  passes `False`, because a conversation-cache write would never be read back.
- **A capability with instructions returns its rules as a module string and its per-run context as
  a callable** (`[RULES, per_run_context]`). Pydantic AI sorts static parts before dynamic ones, so
  every capability's rules form the cached prefix and whatever changes per run follows it.
- **History compaction (the `ProcessHistory` summarizer) is unchanged.** Once a thread is past half
  the context window, each turn rewrites the summarized tool results, so the conversation cache
  misses from the first compacted message onward for the rest of that thread. Accepted; measure
  with `cache_read_input_tokens` in Logfire before changing either side.

## Considered options

| Option | Why not |
| --- | --- |
| Set `anthropic_cache` on each agent's model settings | Provider-specific, and the template is provider-agnostic. `Caching` maps the one setting to whatever the model supports, and a provider-specific setting overrides it entirely when both are set. |
| Keep prompts as one callable and accept caching only the tools | The instructions are most of the prefix; caching the tools alone saves little. |
| Leave caching off until an instance needs it | The setting is a no-op on providers without a unified cache (pydantic-ai honours it on Anthropic, Bedrock, OpenRouter, and OpenAI GPT-5.6 and later), so on by default costs nothing and the convention below holds from the first capability. |

## Consequences

- A capability that adds instructions returns a static string for anything that does not vary per
  run, and a callable only for what does. A literal built with an f-string at import is still static.
- `tests/unit/agents/test_registry.py` checks, through a `FunctionModel`, that the system prompt
  reaches the model as a static part, so a refactor that folds it into a callable fails a test
  rather than silently un-caching the prompt.
- Anthropic ignores a breakpoint on a prefix shorter than the model's minimum (4096 tokens on
  Haiku), so a short prompt on a small model is not expected to hit the cache; the setting is
  harmless there and keeps the recipe uniform.
- Nothing here changes what the model sees; only which parts the provider may serve from cache.
