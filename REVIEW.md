# Review guide

Review checks for CortexDJ: a FastAPI and Pydantic AI backend cut into feature slices
(`chat`, `sessions`, `spotify`, `retrieval`) over a small kernel, and a Next.js chat frontend.
Neither `/code-review` nor `codex review` reads this file on its own: they, and any harness
that loads AGENTS.md, reach it through AGENTS.md's line for review requests. Keep it to what
the built-in reviewers cannot know about this repo — they own methodology, verification, and
severity.

## Skip what the toolchain already enforces

Ruff and ultracite run as edit hooks, pre-commit, and CI; `mypy --strict` runs in pre-commit
and CI; import-linter (the dependency direction) runs in pre-commit and CI; pytest with a
coverage floor runs in CI. Do not spend findings on style, formatting, lint, type
annotations, or "add a test for coverage's sake". Review logic, design, and conventions.

## Rules to check against

Read the rules for the stack the diff touches; a change that breaks one is a finding, and the
finding quotes the rule:

- Backend: `.agents/rules/backend/code-conventions.md`; agent, tools, or evals: `.agents/rules/backend/pydantic-ai.md`.
- Frontend: `.agents/rules/frontend/code-conventions.md`; chat UI or streaming: `.agents/rules/frontend/vercel-ai-sdk.md`.
- Net-new code in any stack: `.agents/rules/conventions.md`.

## Review checks

The conventions the code already shows. A finding quotes the check it breaks; the exceptions
named in a check are never findings.

### Any stack

- **Names say what the code does.** Functions and behavioral modules are verb-first and specific (`generate_thread_title`, `analyze_session`, `encode_session_to_clap_space`, `build_mood_playlist`), never `process`, `handle`, `manage`, or `run`. Exception: framework-convention names stay as they are: `deps.py`, `hooks.py`, `app.py`, a slice's role files (`model.py`, `schema.py`, `service.py`, `router.py`), and React event handlers named `handleX` (`handleSubmit`, `handleDelete`).
- **No grab-bag modules.** No `utils.py`, `helpers.py`, `misc.py`, or `common.py`: code with no home names a missing concept. Exception: `frontend/lib/utils.ts`, shadcn's home for `cn()`.
- **Docstrings carry intent.** A module or public entry point whose purpose isn't obvious from its name has a docstring: one line on what it is, the non-obvious why, the ADR in `docs/adr/` that holds the rationale, and a "do not" wherever the code looks fixable but isn't. It never restates the name or the signature. A new non-obvious decision is the next numbered ADR, not a paragraph in `AGENTS.md`.
- **Docstrings hold slow-changing content**: contracts, invariants, rationale, not step-by-step logic that drifts out of sync. A comment gives the non-obvious why at one line; it never restates what the line does.
- **Docs describe the current state.** Comments, docstrings, ADRs, and eval notes carry no ticket numbers, scratch paths, "previously" / "used to" / "was changed to" narrative, or "amended" blocks. A rejected alternative is described by what it gets wrong, not as what the code did before. Exception: `docs/evals/baseline.md`, whose sections are dated runs kept as recorded.

### Backend

- **Fake only at the seams** in `.agents/rules/backend/code-conventions.md` (Testing). Flag a test that patches any other `cortexdj` function or module global (`monkeypatch.setattr`), and name the seam the source should grow instead: a FastAPI dependency, an `override` context manager, or a constructor argument.
- **Fakes live with their feature.** A fake more than one test file uses goes in the feature's own `tests/integration/features/<name>/conftest.py`, never the shared `tests/integration/conftest.py`, which imports no feature code. A double only one test uses stays in that test.
- **Settings a test reconfigures are read at run time.** The Settings seam reaches only code that calls `get_settings()` when it runs; a module-level `config = get_settings()` is read once at import. Flag a module-level read in code a test must reconfigure, and one without a comment saying why it is read at import and not to copy it.
- **Tests assert on what a caller can observe** (a return value, a response, stored state read back), with expected values written as literals. Exception: an integration test may go below the API (two real connections racing on the `engine` fixture, spans from the app's own engine) with a comment saying why no endpoint can observe it.
- **Test data comes from the builders** in `tests/factories.py` (`make_*` / `create_*`); a shape two test files need gets a builder there.
- **Logging uses f-strings**: `logger.info(f"Created {item.id}")`.
- **No `Args:` / `Returns:` sections** that repeat the type annotations. Exceptions: agent tools, where Pydantic AI sends `Args:` to the model, and router endpoint docstrings, which stay even when short because FastAPI surfaces them in OpenAPI.
- **Shared resources arrive by dependency injection.** A route reaches a DB session or a loaded model or client through a FastAPI dependency; a resource loaded at startup is reached through typed dependency injection, never imported in a route. The two loaded models live in holders (`features/sessions/eeg_model.py`, `features/retrieval/encoder.py`): a route takes one as a dependency (`ContrastiveEncoderDep`), a tool reads the getter, and both pass it to the service as an argument, so a service never reads a holder.
- **Another slice's data arrives through a `shared/` seam.** `spotify` reaches tracks only through `shared/track_access.py`, and `retrieval` reads a session's participant only through `shared/session_lookup.py`. Flag a slice that copies another slice's query or model to get around the import rule, and a seam that hands out an ORM row where a frozen dataclass or a plain value would do.
- **The Spotify client comes from the getter.** Tools, the capability's `prepare_tools`, and scripts call `get_spotify_client(db)` (`features/spotify/client.py`) and branch on `is_user`; flag a client built anywhere else or carried on `AgentDeps`.
- **Vector search stays in the retrieval slice.** Similarity queries over `track_audio_embeddings` go through `TrackAudioEmbedding.search_similar`, ordered by `cosine_distance` so the HNSW index applies; flag a vector query written anywhere else. The index is declared on the model and created with `op.create_index`, never raw SQL.
- **A router imports its slice's service under an alias**: `from cortexdj.features.sessions import service as session_service`, then `session_service.list_sessions(...)`.
- **Blocking secondary work in a route runs as `BackgroundTasks`.** Exception: work that needs a row the request has locked runs as a tracked `asyncio.create_task`, since `BackgroundTasks` run before the request's session commits and would deadlock (title generation in `features/chat/stream.py`).
- **Agents come from the registry.** Every tool-using agent is assembled with `build_agent` (`agents/assistant.py`), and every agent is obtained by name through `get_agent` (`agents/registry.py`). Flag a call site that calls a builder. Exception: a test of the builder itself, since `get_agent` returns the agent cached for the process.
- **Packages stay lean.** A slice's `__init__.py` re-exports its models only, never a router or a service, so `import cortexdj.features` registers every table for Alembic; `features/__init__.py` re-exports every slice's models. Every other `__init__.py` is empty, and code imports deeply (`from cortexdj.features.sessions.model import Session`).
- **Models lean on type inference**, with `mapped_column` only where a column needs customizing. DB logic is in model classmethods named verb-first: `create`, `get` (by primary key) or `get_by_*`, `list_*` (`list_all` for every row), `update_*`, `delete_*`, `search_*`, or a more specific verb where one fits (`append_messages`, `get_or_create`, `touch`).

### Agent runtime (Pydantic AI)

- **Tools that share the session run in sequence.** A capability whose tools share `AgentDeps.db` builds its toolset with `FunctionToolset(sequential=True)`; the reason is in the comment at that line in `features/sessions/capability.py`.
- **Tools raise.** A tool body raises an error rather than returning an error dictionary.
- **A tool that cannot work is hidden.** A tool that needs something a run may not have (a loaded checkpoint, the user's Spotify login) is dropped in its capability's `prepare_tools`, as `get_model_info` and the four library tools are, so the model is never offered a call that can only fail.
- **A tool that writes to the user's Spotify account waits for a yes.** `build_mood_playlist` and `add_tracks_to_playlist` take `user_confirmed` and do nothing without it; flag a new writing tool with no such gate.

### Frontend

- **A slice exports its tool-to-panel map from `tool-panels.tsx`** (`SESSIONS_TOOL_PANELS`) and registers it in `features/tool-panel-registry.ts`. A new panel needs no edit to `message.tsx`, which looks panels up in the registry by tool name.
- **A slice reaches the chat header through `features/header-slot-registry.ts`.** `chat-header.tsx` renders the registry's slots and never imports a slice; a slot component takes the thread id and fetches its own data through the slice's `use-*.ts` hook.
- **The shadcn component over the raw markup**: `<Skeleton>` over `animate-pulse` placeholder blocks, `<Tooltip>` over `title=` on an interactive element, `<Separator>` for a standalone divider, and `<Card>` for a rounded, bordered `bg-card` box whatever its padding or text size; likewise `<AlertDialog>`, `<DropdownMenu>`, `<Collapsible>`, `<Sheet>`, and `<Alert>`. Exceptions: a hidden file input, a non-text input like `type="range"`, `title=` as a truncation hint on text, a border that is part of a container's layout, a pulsing status label like "Generating..." (text, not a placeholder), and `bg-card` on a surface that isn't a rounded, bordered box (a sticky bar, a full-bleed band). A raw `button`, `input`, `textarea`, or `hr` in an exception carries a `biome-ignore` that gives the reason.
- **Use the variant.** Where a shadcn variant matches, the code uses it; `className` overrides only what the variant doesn't cover.
- **`components/ui/` changes only to add a shadcn component or customize a variant.**
- **Memoize only where it pays.** A new `useCallback` or `memo` serves a memoized child, a hook dependency, or a callback a custom hook returns (`scrollToBottom` in `use-scroll-to-bottom.tsx`); flag one added speculatively.
- **A query hook lives with its scope**: a domain-specific hook in its slice (`features/<name>/use-<name>.ts`), a cross-cutting or chat hook under `api/hooks/`.

### Chat UI

- **`ChatMessage` is threaded through.** `useChat` and every `UseChatHelpers` site carry the `ChatMessage` type from `frontend/lib/types.ts`, the one place to type message metadata, data parts, and tool parts; a site on plain `UIMessage` won't see a type added there later.

## Maintainability: flag, don't block

Behavior-preserving simplifications, helpers duplicated across slices that belong in the
kernel, and files grown large for their role are nits, never blockers. This repo prefers
readability over cleverness and asks before architectural changes, so propose rather than
demand. Report at most five nits. On a re-review of the same branch, report only findings
above nit level.

### Smell baseline (nit tier)

Fowler's code smells (_Refactoring_, ch. 3), adapted from Matt Pocock's `code-review` skill. Each
is a labelled heuristic ("possible Feature Envy"), never a hard violation; a rule or review check
above overrides it; skip anything tooling already enforces. Read as *what it is → how to fix*:

- **Mysterious Name**: a name that doesn't reveal what it does or holds → rename; if no honest name comes, the design is murky.
- **Duplicated Code**: the same logic shape in more than one hunk or file → extract the shared shape.
- **Feature Envy**: a method reaching into another object's data more than its own → move it onto the data it envies.
- **Data Clumps**: the same few fields or params travelling together → bundle them into one type.
- **Primitive Obsession**: a primitive standing in for a domain concept → give the concept its own small type.
- **Repeated Switches**: the same type-dispatch cascade recurring across the change → polymorphism, or one shared map.
- **Shotgun Surgery**: one logical change forcing scattered edits across many files → gather what changes together.
- **Divergent Change**: one module edited for several unrelated reasons → split so each changes for one reason.
- **Speculative Generality**: abstraction or hooks for needs the spec doesn't have → delete; inline until a real need shows.
- **Message Chains**: long `a.b().c().d()` navigation → hide the walk behind one method on the first object.
- **Middle Man**: a class or function that mostly delegates onward → cut it, call the target directly.
- **Refused Bequest**: a subclass ignoring most of what it inherits → drop the inheritance, use composition.

## Verdict

End the review with one line: **Ship it**, **Fix and ship**, or **Needs changes**. Only
correctness bugs, security holes, and real convention or architecture breaches gate the
verdict. Nits and pre-existing issues never do.
