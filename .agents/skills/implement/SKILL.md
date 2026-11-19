---
name: implement
description: "Implement a piece of work based on a spec or set of tickets."
disable-model-invocation: true
---

Implement the work described by the user in the spec or tickets. Before starting, apply the implementation-readiness gate in `docs/agents/issue-tracker.md` if that file exists (the `setup-matt-pocock-skills` skill writes it); without it, there is no gate.

If the user passes a ticket reference, fetch it from the issue tracker and state its title before starting. If the reference is ambiguous, ask.

Call the Skill tool with "tdd" where possible, at pre-agreed seams.

Run typechecking regularly, single test files regularly, and the full test suite once at the end.

Commit your work to the current branch as you go.

Once done, review with the `review-changes` skill, passing a mode explicitly
(`solo` unless the user asked for `cross-model`). Fix everything above nit
and commit the fixes.
