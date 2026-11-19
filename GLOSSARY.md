# CortexDJ

A full-stack AI-app template: a FastAPI + Pydantic AI backend and a Next.js chat frontend, with a Notes example domain to copy. An instance replaces this paragraph and these sections with its own domain; the `domain-modeling` skill adds and sharpens entries as terms get settled.

## Language

### Code layout

**Slice**:
One domain's folder under `features/`, holding its model, schemas, service, routes, and (if it talks to an agent) its tools and capability. A slice imports the kernel and never another slice.
_Avoid_: Module, feature folder, app (Django sense)

**Kernel**:
The code every slice shares: `core/` (settings, database, startup hooks, logging, observability), `shared/` (base classes, scrubbing), and `agents/` (the agent runtime). It never imports a slice.
_Avoid_: Framework, platform, common

**Composition root**:
One of the few files allowed to import across slices because its job is to wire them together: `api.py` (routers), `agents/capabilities.py` (the assistant's capabilities), `agents/registry.py` (named agents), and the frontend's `features/tool-panel-registry.ts`.
_Avoid_: Registry (the named-agent registry is one composition root, not the pattern), index, barrel

**Seam**:
A boundary the source declares so one thing can be swapped for another: a FastAPI dependency, an `override` context manager, a constructor argument, or a registration slot the composition root fills. Tests fake only at seams.
_Avoid_: Mock point, hook (reserved for startup hooks and agent hooks), patch target

**Feature marker**:
An `# sf:begin(<id>)` / `# sf:end(<id>)` block or `# sf:keep-if(<id>)` line that tells the onboarding helper which code belongs to a cuttable feature.
_Avoid_: Flag, feature toggle (nothing is toggled at runtime)

**Rule**:
A convention an implementer needs before writing code: one the existing code doesn't show and no tool or test would catch. Rules are scoped to the paths they govern, so an agent reads only the ones for the files it touches.
_Avoid_: Guideline, check, convention (a rule is the subset of conventions that can't be pulled from the code)

**Review check**:
A convention a reviewer applies to a diff: one the code already shows or that is cheap to fix after the fact. Implementers don't load review checks.
_Avoid_: Check (alone; it also means a toolchain run or a deterministic check), review rule

### Chat

**Thread**:
One conversation, identified by the id the browser chose, with its stored history and a title.
_Avoid_: Chat, session, conversation (fine in prose, not as the noun for the row)

**Turn**:
One user message and the run that answers it. The browser sends only the newest message; the server owns the history.
_Avoid_: Request (the HTTP request is one part of it), exchange, round

**Thread history**:
The stored messages of one thread: what the model is handed as `message_history` on the next turn. The stored copy is what the model saw, scrubbed and sanitized.
_Avoid_: Conversation, context window, transcript

**Regenerate**:
The browser re-sending a user turn the server stored but never answered (the Retry on a failed reply). The stored copy stands in; the turn is not stored twice.
_Avoid_: Retry (the button's label; the mechanism is a regenerate), resend

**Tool panel**:
A frontend component mounted beneath a tool call's row when the tool's name is in the tool-panel registry, rendering the call's input or output in the domain's own shape.
_Avoid_: Widget, card (one kind of panel), rich tool view

### Agents

**Capability**:
A Pydantic AI capability a slice owns (`features/<name>/capability.py`): its toolset, its instructions, and any hooks. The assistant is the sum of its enabled capabilities on the shared recipe.
_Avoid_: Plugin, toolset (the toolset is one part of it), skill

**Shared recipe**:
`build_agent` in `agents/assistant.py`: the runtime wiring every named agent gets (tool-error hooks, history compaction, prompt caching, optional thinking, the tool timeout). An agent adds its capabilities and output type and inherits the rest.
_Avoid_: Base agent, agent factory, template agent

**Named agent**:
An agent the registry builds lazily by name (`get_agent(ASSISTANT)`). Callers ask for a name and never import a builder.
_Avoid_: Agent instance, singleton

### Sensitive data

**Sensitive identifier**:
A card number, Social Security number, or email address in user text, which the assistant never needs to do its job.
_Avoid_: PII, personal data, secret

**Scrubbing**:
Replacing every sensitive identifier in a piece of text with a placeholder before anything stores it, sends it to a model, or traces it. Masking and redaction are its two treatments.
_Avoid_: Sanitizing (the adapter's cleanup of untrusted client messages), filtering, PII detection

**Masking**:
Scrubbing that keeps the last four digits so the identifier can still be referred to. Used for cards and Social Security numbers.
_Avoid_: Partial redaction, truncation

**Redaction**:
Scrubbing that keeps nothing of the identifier, only a placeholder naming its kind. Used for emails. Also what the request log and request span do to credential query parameters.
_Avoid_: Removal, tokenization

### Retrieval

**Hybrid search**:
Searching the notes two ways at once, by meaning and by wording, and merging the two rankings into one. Every note search is a hybrid search; there is no mode that runs only one leg.
_Avoid_: Semantic search (one leg of it), vector search, RAG

**Semantic leg**:
The half of a hybrid search that ranks notes by how close their meaning is to the query's, using the stored embeddings. It finds a note that uses none of the user's words, and never a note that has no embedding.
_Avoid_: Vector leg, dense retrieval, similarity search

**Keyword leg**:
The half of a hybrid search that ranks notes by the words they share with the query, after both are reduced to word stems and common words are dropped. A match in a note's title counts for more than a match in its content. It finds a note that names the exact thing the user asked about, embedding or not.
_Avoid_: Full-text leg, lexical search, BM25 (Postgres ranks differently)

**Reciprocal rank fusion**:
How a hybrid search merges its two rankings: each leg that returned a note adds one over sixty plus the note's rank in that leg, and the notes are ordered by the sum. Only ranks count, so a note both legs found beats one only a single leg found. The sum is the result's score; it is not a similarity and means nothing outside the search that produced it.
_Avoid_: RRF score as a similarity, blending, reranking

### Testing and evaluation

**Test tier**:
One of the three test directories, each its own run: unit (no database, no provider), integration (real Postgres and HTTP), evals (real models, costs money).
_Avoid_: Test level, suite, marker (a tier is a directory, not a marker)

**Judge**:
A model-graded check: one rubric, one behavior, pass or fail with a reason, run on `EVAL_JUDGE_MODEL`, which is pinned a tier above the agents' model.
_Avoid_: Grader (the golden set's term for the callable), LLM-as-judge, scorer

**Rubric**:
A statement that is true of a good reply, which a judge grades a reply against. One rubric names one behavior, so a failing row says which behavior broke.
_Avoid_: Criteria, prompt (the rubric is rendered into one), checklist

**Deterministic check**:
An evaluator that is a plain function of the output or the recorded tool calls, with no model call. Run before judges: it is free and never disagrees with itself.
_Avoid_: Assertion, unit check, heuristic

**Golden set**:
Hand-written replies, each labeled good or bad for one judged behavior, used to check that a judge grades the way a person would. It tests the judges, not the agent.
_Avoid_: Calibration dataset, judge tests, ground truth

**Baseline run**:
The agent cases at a high repeat count, recorded as a dated section in `docs/evals/baseline.md`, before and after a deliberate prompt or model change. A measurement to compare against, not a gate.
_Avoid_: Benchmark, regression suite, threshold
