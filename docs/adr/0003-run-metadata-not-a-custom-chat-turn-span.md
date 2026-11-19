# 0003: Run metadata on the agent span, and nothing the libraries already record

**Status:** accepted (2026-10-17)

## Context

The standard Logfire integration (`logfire.configure` plus the FastAPI, SQLAlchemy, and Pydantic
AI instrumentors, with nothing exported unless `LOGFIRE_TOKEN` is set) already records most of
what custom spans, attributes, and trace relationships would add.

Every Pydantic AI run span carries `gen_ai.agent.name`, `gen_ai.agent.call.id`, and
`gen_ai.conversation.id` (the thread id, passed as `conversation_id`), and those propagate as
OTel baggage onto every tool and model request span under the run; tool spans record arguments,
results, and exceptions; model spans record messages and usage; `Dataset.evaluate` opens an
experiment span with the dataset name, pass rate, and one child span per case.

What the baseline cannot say is anything the deps carry that the instrumentors cannot see: in an
instance, who the run was for, or which rule routed it to this agent.

## Decision

- **Facts about a run that a trace should be searchable by go on the run as `metadata`**, set on
  the agent (a callable of `RunContext` reading `ctx.deps`) rather than per call, so every entry
  point that runs the agent through the registry carries them. Pydantic AI records run metadata
  on the `invoke_agent` span as a `metadata` JSON attribute when the run ends, on success and on
  failure, next to the agent name and the thread id the span already carries. The template's
  `AgentDeps` has nothing to record yet; the first instance fact goes there, not in a span.
- **`metadata` on `Dataset.evaluate`** (model, provider, git sha) so experiments are comparable
  across model swaps. No `name=`: the experiment span already records `dataset_name`.
- **Content capture stays on** (Pydantic AI's default): the content is what makes a trace
  readable. Real user data flips this off or adds scrubbing; that is a settings change, not a
  redesign.
- **Nothing else.** In particular:
  - No span of our own around the chat turn. The adapter's `streaming_response` is lazy, so the
    agent runs after the handler returns; a span held open across that boundary ends late, in
    another context, and fails on every Stop click or dropped connection.
  - `agents/hooks.py::on_tool_execute_error` keeps its stdlib `logger.exception`. Pydantic AI
    already records the exception on the tool span; a `logfire.exception` there is a duplicate.
  - No OTel baggage or `Instrumentation` capability to push run facts onto child spans. Filtering
    tool spans by them is not a question anyone has asked; baggage is a one-line upgrade if it
    becomes one.
- **Acceptance is a person looking at the Logfire UI**, not a span count. The in-memory
  `capfire` tests only guard the span shape (one connected trace per turn, the thread id on the
  run) against regressions.

## Considered options

| Option | Why not |
| --- | --- |
| A `chat turn` span of our own around the turn | Fights the lazy streaming response: closes early around the handler, and around the stream needs a response subclass, context plumbing, and error marking, then still fails the span on every Stop click. Run `metadata` puts the same facts on `invoke_agent` with one argument. |
| Baggage for run facts | Reaches every child span, but needs context-manager plumbing for a question nobody is asking. |
| Per-agent `Instrumentation` capability with custom attributes | Pydantic AI-native, but attributes are fixed at agent build time and the interesting facts are only known per run. |
| `logfire.exception` in the tool error hook | Duplicate of what the tool span already records. |
| `include_content=False` in production | Correct for real data, but hides the prompts and tool results that make a template's traces worth reading. Recorded as the swap. |

## Consequences

- A trace for one turn reads: `POST /api/agent/chat` → `invoke_agent assistant` → model requests
  and tool calls, with SQL alongside. `invoke_agent` is where to look for who and why (its name,
  `gen_ai.conversation.id`, and `metadata` once an instance sets it); its children are what happened.
- `core/observability.py` stays the one entrypoint and holds no span helpers; `stream.py` says
  not to wrap the turn.
