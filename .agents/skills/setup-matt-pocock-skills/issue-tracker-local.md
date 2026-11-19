# Issue tracker: Local Markdown

Issues and specs for this repo live as markdown files in `.scratch/`. This guide is the project's workflow contract: the vendored skills say "the issue tracker", and this file says what that means here. Work runs: planning, then the user's manual `to-spec`, `to-tickets` when useful, `implement` or `implement-spec`, review, the user's approval to merge, then cleanup.

## Conventions

- One effort per directory: `.scratch/<effort>/`
- The spec, when there is one, is `.scratch/<effort>/spec.md`
- Implementation issues are one file per ticket at `.scratch/<effort>/issues/<NN>-<slug>.md`, numbered from `01`, never a single combined tickets file. Supporting files (run logs, transcripts) go in `.scratch/<effort>/evidence/`
- A file stays where it is for its whole life; status changes never move it
- Refer to a ticket in another effort by effort and number (`<effort> 09`), since numbers repeat across directories
- Triage state is recorded as a `Status:` line near the top of each issue file (see `triage-labels.md` for the role strings when `triage` is installed)
- Comments and conversation history append to the bottom of the file under a `## Comments` heading

A new effort numbers its tickets from `01`; a ticket added to an existing effort takes the next number after the highest in that directory, even where a directory starts above `01` or has gaps. Never renumber.

## Status

Specs and implementation tickets carry one of:

- `ready-for-agent`: eligible once its hard blockers are resolved (see Implementation readiness)
- `needs-planning`: waits for a planning session
- `resolved (YYYY-MM-DD, PR #N)`: reviewed, and the user approved merging. A spec finished across several PRs lists them (`PRs #49 to #51`). An older entry that lacks the date or PR is filled in only from history, never by guessing

Any status that starts with `resolved` counts as resolved for blockers and the index. A spec resolves with its last ticket, or on its own when implemented directly.

## Active work

`.scratch/active.md` links every unresolved implementation ticket, `needs-planning` ones included. Creating a ticket adds its link; resolving it removes the link. It is an index only: each ticket's own `Status:` and `Blocked by:` lines decide eligibility, so the index never repeats them. Wayfinder decision tickets are not listed.

## Implementation readiness

Implementation tickets are eligible only when `Status: ready-for-agent` and every hard blocker is resolved. `needs-planning` excludes a ticket from implementation even when its blockers are resolved or an implementation request names the wider spec. Report the planning gate and continue other eligible work; a named ticket still needs its planning session before implementation.

For a ticket that needs a design session, use:

```markdown
**Status:** needs-planning
**Planning approach:** grill-with-docs (or wayfinder)

**Planning gate:** Refine against the current implementation with the user before
implementation. Update scope, acceptance criteria, and dependencies; record
settled domain terms and architectural decisions where needed. Return to
ready-for-agent once the user agrees the plan is settled.
```

The existing build description and acceptance criteria are provisional while a ticket needs planning. Resolve its open questions against the code that has landed, then rewrite the ticket to reflect the agreed result. Use `to-spec` if a separate spec is useful and `to-tickets` if the work needs splitting. Preserve existing ticket IDs; add new IDs rather than renumbering unrelated work. Link planning artifacts from the original ticket. Update the glossary and ADRs for decisions actually made, not hypothetical designs.

`Planning approach` recommends the session format; the marker alone does not launch that skill. Wayfinder decision tickets use the separate lifecycle below. Completing a map does not automatically make its implementation ticket ready: incorporate the decisions and obtain the user's agreement first.

Preferred sequence and nice-to-have relationships are not hard blockers.

## Planning handoff

Grill Me, Grill with Docs, and Wayfinder end planning with two lists, the settled decisions and the remaining uncertainties, then stop. Do not offer to write a spec or tickets, and do not start `to-spec` or `to-tickets`: the user invokes them. When Wayfinder finds no fog and its skill says to ask how to proceed, this summary and stop is that answer. Wayfinder still creates, claims, and resolves its decision tickets during planning, as Wayfinding operations below describes.

## Completing work

This tracker closes work through PRs when the repo has a remote. In a repo with no remote, skip the PR: on the user's approval, set the status to `resolved (YYYY-MM-DD)`, merge the branch into `main` locally, and apply the cleanup below against `main` instead of `origin/main`, with no remote branch to delete.

1. **Resolve on approval.** After review, wait for the user's explicit approval to merge. Then, before merging, set each ticket the PR completes (and its spec, if this finishes it) to `resolved (YYYY-MM-DD, PR #N)`, remove those tickets from `.scratch/active.md`, and commit and push that to the PR branch, so the PR carries its own bookkeeping.
2. **Merge.** If the merge fails (conflicts, failing checks, permissions), report the failure and its error plainly. The resolution records reviewed, approved work; it does not mean the merge happened, so never say it did.
3. **Clean up after a successful merge.** First check there is nothing left to lose: `git fetch`, then `git merge-base --is-ancestor <branch> origin/main` must succeed, and the worktree must have no uncommitted changes. (`git branch -d` alone is not that check: it compares against the branch's upstream, not `main`.) Then remove the worktree, with the environment's worktree tool when the session created it, otherwise `git worktree remove`; then delete the branch locally (`git branch -d`) and on the remote. Leave everything in place when the check fails, or the branch is a research or prototype branch or unrelated to this work. Review branches stay until their merge succeeds.

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<effort>/` (creating the directory if needed). For each new implementation ticket, add a link to `.scratch/active.md`.

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. The user will normally pass the path or the issue number directly.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a file with one **child** file per ticket.

- **Map**: `.scratch/<effort>/map.md` (the Notes / Decisions-so-far / Fog body).
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the question in the body. A `Type:` line records the ticket type (`research`/`prototype`/`grilling`/`task`); a `Status:` line records `claimed`/`resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when every file it lists is `resolved`.
- **Frontier**: scan `.scratch/<effort>/issues/` for files that are open, unblocked, and unclaimed; first by number wins.
- **Claim**: set `Status: claimed` and save before any work.
- **Resolve**: append the answer under an `## Answer` heading, set `Status: resolved`, then append a context pointer (gist + link) to the map's Decisions-so-far in `map.md`.
- **Implementation tickets** a map leads to live in an implementation effort's directory, never in the map's `issues/`, so the frontier scan sees only decisions. The Status and Completing work rules above apply to them, not to decision tickets.
