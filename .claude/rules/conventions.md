---
paths:
  - "backend/**/*.py"
  - "frontend/**/*.{ts,tsx}"
---

# Code Conventions

How to write code here so an agent can find it by `grep` and read its intent
without guessing. These codify existing practice and guard against drift — they
introduce nothing new and touch no existing code; expand as patterns recur.
Per-stack mechanics (filename casing, typing, imports) live in
`.claude/rules/{backend,frontend}/code-conventions.md`; this file is the
cross-cutting, language-agnostic layer.

## Names are how agents find code

Agents `grep` for what they're trying to do before they read. Name for behavior,
not layer.

- **Functions and behavioral modules:** verb-first and specific —
  `generate_thread_title`, `encode_session_to_clap_space`,
  `retrieve_similar_tracks`, `build_mood_playlist`. Not `process`, `handle`,
  `manage`, `run`.
- **No grab-bag modules.** Never `utils.py`, `helpers.py`, `misc.py`,
  `common.py`. Code with no home names a missing concept — find it. (Frontend
  exception: `lib/utils.ts` is shadcn's sanctioned home for `cn()`; keep it.)
- **Keep framework-convention names.** `deps.py`, `hooks.py`, `app.py`, and the
  `models/`, `schemas/`, `routers/` layer dirs are what an agent expects in a
  FastAPI / Pydantic AI / SQLAlchemy repo — discoverability comes *from* the
  convention. Don't "behavioralize" these.

## Docstrings carry intent; comments carry local "why"

An agent retrieves a symbol, then reads its docstring — so the docstring is the
contract. Every module and public entry point gets one (leaf functions stay lean
per the per-stack rules; don't restate a signature). The recipe (see
`backend/src/cortexdj/services/retrieval.py` for the model):

1. One line: what it is.
2. The non-obvious *why* — the thing an agent would otherwise get wrong.
3. Cross-ref the governing decision: an ADR once `docs/adr/` exists, else the
   rules file / migration comment that holds the rationale today.
4. A **"do not"** wherever the code looks fixable but isn't — e.g.
   `ml/dataset.py`'s *"Don't 'optimize' this loop away"*, or
   `agents/tools/retrieval_tools.py`'s intentional inline `DeapFileMissingError`
   catch (every other tool lets exceptions propagate to `hooks.py`). Agents act
   on negative constraints; state intentional weirdness explicitly or it gets
   "fixed."

Agent **tool** docstrings double as the runtime LLM tool description (Pydantic
AI), so usage caveats and "do not"s there are load-bearing — see
`build_mood_playlist`'s *"You must get explicit user confirmation before calling
this tool."*

Keep docstrings to slow-changing content (contracts, invariants, rationale), not
step-by-step logic that drifts out of sync. Inline comments only for non-obvious
*why* at a specific line — never to restate *what*.

## Rationale lives in ADRs, not CLAUDE.md

`AGENTS.md` (CLAUDE.md is its symlink) = invariants + entry points + pointers;
keep it lean (ceiling ~200 lines). The *why* behind a non-obvious choice belongs
in `docs/adr/` — and the docstring that implements it cites the ADR, so an agent
meets the rationale before it "simplifies" the choice away.

`docs/adr/` doesn't exist yet; it lands via a `grill-with-docs` pass that will
promote today's scattered rationale (HNSW-over-IVFFlat in the pgvector
migration, iTunes-over-`preview_url` in `services/audio_catalog.py`, the
two-layer tool-error convention, label binarization at the subject median) into
numbered records. Until then, cite the existing home and keep new rationale out
of `AGENTS.md`.
