# Pydantic AI Rules and Guidelines for LLMs

This file provides rules and guidelines for LLMs when working with Pydantic AI code in cortexdj.

## Docs are split across two places — know which one to consult

Pydantic AI's documentation lives in two places, and as of April 2026 they contain **different content**:

1. **`docs/pydantic-ai-llms-full.txt`** — the local pinned snapshot from `https://ai.pydantic.dev/llms-full.txt`. Refresh with:

   ```bash
   curl -sSL https://ai.pydantic.dev/llms-full.txt -o docs/pydantic-ai-llms-full.txt
   ```

   This file is now **reference-only**. It contains:
   - Inline API reference for every `pydantic_ai.*` module (agent, capabilities, messages, tools, toolsets, models, providers, run, settings, ui, …)
   - Model provider API docs (anthropic, google, openai, mistral, bedrock, etc.)
   - Three front-loaded "advanced features": Image/Audio/Video Input, HTTP Request Retries, Thinking
   - The full `pydantic_evals` surface (datasets, evaluators, lifecycle hooks, logfire integration)
   - Agents section (condensed)

2. **`https://ai.pydantic.dev/`** (web docs, fetch ad-hoc with WebFetch) — contains:
   - All conceptual guides and tutorials
   - Example applications: chat-app-with-FastAPI, RAG, Data Analyst, DuckDB SQL, Slack Lead Qualifier, Flight Booking multi-agent
   - Durable execution walkthroughs (DBOS / Prefect / Temporal)
   - Beta Graph API guide
   - Multi-page Agent2Agent (A2A / FastA2A) protocol guide
   - Introduction, installation, getting help, troubleshooting

**Rule of thumb:** if you're grepping `llms-full.txt` for a worked example or a learning walkthrough and finding nothing, the content hasn't been deleted — it lives at `https://ai.pydantic.dev/` now. Use WebFetch before giving up. The `updating-deps` skill refreshes `llms-full.txt` only; web guides are not cached locally.

Prior to the April 2026 restructuring, `llms-full.txt` contained both reference and tutorial content in a single 168k-line file. Older snapshots in git history will show tutorial content that the current file no longer has.

---

## Cortexdj-specific pydantic-ai usage

These are facts about how cortexdj wires pydantic-ai that aren't derivable from generic pydantic-ai docs:

- **Agent entrypoint:** `backend/src/cortexdj/agents/brain_agent.py`. Constructs `brain_agent = Agent(...)` with `OpenAIResponsesModel`, `AgentDeps` as the deps type, four custom `AbstractCapability` subclasses, a `Hooks` instance (see `agents/hooks.py`), and `history_processors=[summarize_tool_results]`.

- **Deps shape:** `backend/src/cortexdj/agents/deps.py` — `AgentDeps` is a dataclass carrying `db` (async SQLAlchemy session), `eeg_model`, `spotify_client`, `thread_id`, and `brain_context`. Every tool receives `ctx: RunContext[AgentDeps]` and reads `ctx.deps.*`.

- **Capabilities live in `backend/src/cortexdj/agents/capabilities/`.** Four of them: `SessionCapability`, `InsightCapability`, `PlaylistCapability`, `ClassificationCapability`. Each overrides `get_toolset()` to return a `FunctionToolset`. `PlaylistCapability` and `ClassificationCapability` also override `prepare_tools()` to filter tools at runtime based on whether Spotify is connected / an EEG model is loaded. `ClassificationCapability` additionally overrides `get_instructions()` to inject the current brain context into the system prompt.

- **Use `history_processors=` as the native parameter, not `HistoryProcessor(...)` wrapped in `capabilities=[...]`.** Both work, but the direct parameter is the idiomatic form and is what every example in the upstream docs uses.

- **Tool-failure recovery goes through hooks.** See `backend/src/cortexdj/agents/hooks.py` — `on_tool_execute_error` catches exceptions raised inside tool bodies and returns a structured error dict so the agent can respond conversationally instead of crashing the stream. When adding a new tool that makes external calls, prefer letting exceptions propagate up to the hook rather than catching them inside the tool.

- **Large tool results are summarized in prior turns** by `backend/src/cortexdj/agents/history_processor.py` (`summarize_tool_results`). The current turn's tool returns are kept full-sized; historical returns for Spotify/EEG list-style tools are replaced with `{_summarized: True, count, sample, total_available}` to prevent token bloat. The summarizable tool list is `SUMMARIZABLE_TOOLS` at the top of that file.

- **Streaming endpoint:** `backend/src/cortexdj/routers/agent.py` uses `VercelAIAdapter.dispatch_request(..., sdk_version=6)` with an `on_complete` callback for persistence. Do not hand-roll SSE events — the adapter handles the Vercel AI SDK protocol.

- **Logfire is wired** via `logfire.configure(service_name="cortexdj")` + `logfire.instrument_pydantic_ai()` at import time in `brain_agent.py`. This auto-traces every agent run, every tool call, every model request. No manual spans needed.

- **Evals live in `backend/tests/evals/`** and are gated behind the `@pytest.mark.eval` marker so they don't run in the default `pytest` invocation. Run them explicitly with `uv run --directory backend pytest -m eval`. Use `TestModel` or `FunctionModel` for deterministic assertions that don't hit the OpenAI API; reserve real-model runs for manual validation.

## Reference

- Pinned local reference: `docs/pydantic-ai-llms-full.txt`
- Live web guides (tutorials, examples, how-tos): <https://ai.pydantic.dev/>
- API reference browsable: <https://ai.pydantic.dev/api/>
