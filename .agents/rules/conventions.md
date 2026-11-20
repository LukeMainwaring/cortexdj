---
paths:
  - "backend/**/*.py"
  - "frontend/**/*.{ts,tsx}"
---

# Code conventions

The cross-cutting rules; per-stack rules live in `.agents/rules/{backend,frontend}/`, and
everything the code already shows is a review check in `REVIEW.md`.

## Docstrings carry the why

Write the docstring while you still know the why: what it is in one line, then what an agent
would otherwise get wrong, the ADR in `docs/adr/` that holds the rationale, and a "do not"
wherever the code looks fixable but isn't.

An agent tool's docstring is the model's tool description at runtime (Pydantic AI), so say
when to call it rather than a nearby tool (`set_brain_context`: "Call when the user references a
specific session or mood").

## Docs describe the current state

Comments, docstrings, ADRs, and eval notes say what is true now; git holds the history, so
leave out ticket numbers, scratch paths, and "previously" narrative. The one dated log is
`docs/evals/baseline.md`.
