# CortexDJ

A chat app over EEG recordings of people listening to music: one assistant classifies what a listener's brain did, explains it, and curates Spotify playlists and track matches from it. This file holds the words the code and the docs use for that domain and for the app's own structure; the `domain-modeling` skill adds and sharpens entries as terms get settled.

## Language

### Code layout

**Slice**:
One domain's folder under `features/`, holding its model, schemas, service, routes, and (if it talks to an agent) its tools and capability. A slice imports the kernel and never another slice.
_Avoid_: Module, feature folder, app (Django sense)

**Kernel**:
The code every slice shares: `core/` (settings, database, startup hooks, logging, observability), `shared/` (base classes and the seams one slice offers the others), `agents/` (the agent runtime), and `ml/` (EEG model inference). It never imports a slice.
_Avoid_: Framework, platform, common

**Composition root**:
One of the few files allowed to import across slices because its job is to wire them together: `api.py` (routers), `agents/capabilities.py` (the assistant's capabilities), `agents/registry.py` (named agents), and the frontend's `features/tool-panel-registry.ts` and `features/header-slot-registry.ts`.
_Avoid_: Registry (the named-agent registry is one composition root, not the pattern), index, barrel

**Seam**:
A boundary the source declares so one thing can be swapped for another: a FastAPI dependency, an `override` context manager, a constructor argument, or a registration slot the composition root fills. Tests fake only at seams.
_Avoid_: Mock point, hook (reserved for startup hooks and agent hooks), patch target

**Rule**:
A convention an implementer needs before writing code: one the existing code doesn't show and no tool or test would catch. Rules are scoped to the paths they govern, so an agent reads only the ones for the files it touches.
_Avoid_: Guideline, check, convention (a rule is the subset of conventions that can't be pulled from the code)

**Review check**:
A convention a reviewer applies to a diff: one the code already shows or that is cheap to fix after the fact. Implementers don't load review checks.
_Avoid_: Check (alone; it also means a toolchain run or a deterministic check), review rule

### Domain

**Session**:
One EEG recording: one participant listening to a run of stimulus tracks. The assistant and the UI call it by a stable label ("Session 07"), never by its id.
_Avoid_: Recording, sitting; never for a conversation (that is a thread) or a database session

**Segment**:
A 4-second window of a session's EEG, stored with an arousal score, a valence score, band powers, and the quadrant they fall in.
_Avoid_: Window (the model's input, before it is stored), epoch, sample

**Band power**:
The power of a segment's EEG in one frequency band, averaged across channels. The session chart draws each band's share of a segment's total.
_Avoid_: Band energy, spectrum; differential entropy (the log of a band power, which is what the classifier reads)

**Quadrant**:
One of the four emotion states a pair of arousal and valence scores maps to: excited (high arousal, high valence), stressed (high, low), relaxed (low, high), calm (low, low). High is at or above the midpoint: a score of 0.5, or a self-rating of 5 on DEAP's 1 to 9 scale.
_Avoid_: Mood (the word the user and the playlist tools use for a quadrant), emotion class, label

**Brain context**:
The brain state a thread is focused on: a session id, a dominant mood, and average arousal and valence, each optional. It is stored per thread, written into the assistant's instructions on every run, and shown in the chat header.
_Avoid_: Brain state (what a segment or session shows; the context is what the conversation holds on to), session context

**Trajectory**:
A session's path through arousal and valence, one point per track (the mean of that track's segments), with what is computed from it: the share of tracks in each quadrant, transitions (a quadrant change from one track to the next, timed at the start of the track it lands on), centroid, dispersion, and path length. The session card's label is counted over the same tracks; quadrant changes between segments of one track are not transitions.
_Avoid_: Timeline (the per-segment chart), emotional arc, history

**Track index**:
The table of CLAP audio vectors, one per track, that retrieval ranks against a session's EEG embedding by cosine distance.
_Avoid_: Embedding store, catalog, vector database

**Header slot**:
A frontend component a slice registers in the header-slot registry to show a thread's status in the chat header. It takes the thread id and fetches its own data; the brain-context badge is the one slot.
_Avoid_: Header widget, badge (the badge is one slot), status bar

### Chat

**Thread**:
One conversation, identified by the id the browser chose, with its stored history and a title.
_Avoid_: Chat, session, conversation (fine in prose, not as the noun for the row)

**Turn**:
One user message and the run that answers it. The browser sends only the newest message; the server owns the history.
_Avoid_: Request (the HTTP request is one part of it), exchange, round

**Thread history**:
The stored messages of one thread: what the model is handed as `message_history` on the next turn. The stored copy is what the model saw, sanitized.
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
