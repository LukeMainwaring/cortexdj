# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --json number,title,body,labels,comments`.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Make an issue a sub-issue of a parent**: `gh issue create --parent <parent> ...`, or `gh issue edit <parent> --add-sub-issue <child>` afterwards (`gh` 2.94+). Older `gh`: `gh api --method POST repos/<owner>/<repo>/issues/<parent>/sub_issues -F sub_issue_id=<child-db-id>` (database id, as in **Blocking** below). Without sub-issues, put `Part of #<parent>` at the top of the child body.
- **Blocking**: GitHub's **native issue dependencies**, the canonical, UI-visible representation. Add an edge with `gh issue create --blocked-by <n>,<n> ...` or `gh issue edit <child> --add-blocked-by <blocker>` (`gh` 2.94+). Older `gh`: `gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`, where `<blocker-db-id>` is the blocker's numeric **database id** (`gh api repos/<owner>/<repo>/issues/<n> --jq .id`, _not_ the `#number` or `node_id`). Where dependencies aren't available, fall back to a `Blocked by: #<n>, #<n>` line at the top of the child body. A ticket is unblocked when every blocker is closed.
- **Count open blockers**: `gh api repos/<owner>/<repo>/issues/<n> --jq .issue_dependencies_summary.blocked_by` (open blockers only; 0 means startable). `gh issue view <n> --json blockedBy` also lists closed blockers, so filter it: `--jq '[.blockedBy.nodes[] | select(.state == "OPEN")] | length'`.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: post the explanation with `gh issue comment <number> --body "..."`, then `gh issue close <number>`. On an already-closed issue, `gh issue close --comment` drops the comment and still exits 0.

Infer the repo from `git remote -v`; `gh` does this automatically when run inside a clone.

File issues only into the user's own repo. Never file into `mattpocock/skills` (the repo these skills come from) unless the user owns it. In a fork, `gh` can default to the upstream parent, so before the first `gh issue create` of a session, check the target with `gh repo view --json nameWithOwner` and confirm it with the user if it is not their repo.

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature requests; `/triage` reads this flag.)_

When set to `yes`, PRs run through the same labels and states as issues, using the `gh pr` equivalents:

- **Read a PR**: `gh pr view <number>`, then `gh pr view <number> --comments` (off a TTY, `--comments` prints only the comments, not the body), and `gh pr diff <number>` for the diff.
- **List external PRs for triage**: `gh api --paginate 'repos/{owner}/{repo}/pulls?state=open' --jq '.[] | select(.author_association | IN("OWNER","MEMBER","COLLABORATOR") | not) | {number, title, author: .user.login, author_association, labels: [.labels[].name]}'`.
- **Comment / label / close**: `gh pr comment`, `gh pr edit --add-label`/`--remove-label`, `gh pr close`.

GitHub shares one number space across issues and PRs, so a bare `#42` may be either: resolve with `gh pr view 42` and fall back to `gh issue view 42`.

## Status

Specs and implementation tickets carry one of these labels:

- `ready-for-agent`: eligible once its hard blockers are resolved (see Implementation readiness)
- `needs-planning`: waits for a planning session

A ticket is resolved when it is closed. Close it with a comment naming the PR that completed it (`PR #N`); a spec finished across several PRs lists them. A spec resolves with its last ticket, or on its own when implemented directly.

## Implementation readiness

Implementation tickets are eligible only when labelled `ready-for-agent` and every hard blocker is closed. `needs-planning` excludes a ticket from implementation even when its blockers are closed or an implementation request names the wider spec. Report the planning gate and continue other eligible work; a named ticket still needs its planning session before implementation.

For a ticket that needs a design session, label it `needs-planning` and put this at the top of its body:

```markdown
**Planning approach:** grill-with-docs (or wayfinder)

**Planning gate:** Refine against the current implementation with the user before
implementation. Update scope, acceptance criteria, and dependencies; record
settled domain terms and architectural decisions where needed. Relabel
ready-for-agent once the user agrees the plan is settled.
```

The existing build description and acceptance criteria are provisional while a ticket needs planning. Resolve its open questions against the code that has landed, then rewrite the ticket to reflect the agreed result. Use `to-spec` if a separate spec is useful and `to-tickets` if the work needs splitting. Preserve existing ticket numbers; open new issues rather than repurposing unrelated ones. Link planning artifacts from the original ticket. Update the glossary and ADRs for decisions actually made, not hypothetical designs.

`Planning approach` recommends the session format; the label alone does not launch that skill. Wayfinder decision tickets use the separate lifecycle below. Completing a map does not automatically make its implementation ticket ready: incorporate the decisions and obtain the user's agreement first.

Preferred sequence and nice-to-have relationships are not hard blockers.

## Planning handoff

Grill Me, Grill with Docs, and Wayfinder end planning with two lists, the settled decisions and the remaining uncertainties, then stop. Do not offer to write a spec or tickets, and do not start `to-spec` or `to-tickets`: the user invokes them. When Wayfinder finds no fog and its skill says to ask how to proceed, this summary and stop is that answer. Wayfinder still creates, claims, and resolves its decision tickets during planning, as Wayfinding operations below describes.

## Completing work

This tracker closes work through pull requests.

1. **Resolve on approval.** After review, wait for the user's explicit approval to merge. Then, before merging, close each ticket the PR completes (and its spec, if this finishes it) with a comment naming the PR, so the PR carries its own bookkeeping.
2. **Merge.** If the merge fails (conflicts, failing checks, permissions), report the failure and its error plainly. A closed ticket records reviewed, approved work; it does not mean the merge happened, so never say it did.
3. **Clean up after a successful merge.** First check there is nothing left to lose: `git fetch`, then `git merge-base --is-ancestor <branch> origin/main` must succeed, and the worktree must have no uncommitted changes. (`git branch -d` alone is not that check: it compares against the branch's upstream, not `main`.) Then remove the worktree, with the environment's worktree tool when the session created it, otherwise `git worktree remove`; then delete the branch locally (`git branch -d`) and on the remote. Leave everything in place when the check fails, or the branch is a research or prototype branch or unrelated to this work. Review branches stay until their merge succeeds.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Read it as in **Read an issue** above.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body. `gh issue create --label wayfinder:map`.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue (see **Make an issue a sub-issue of a parent**). Where sub-issues aren't enabled, add the child to a task list in the map body and put `Part of #<map>` at the top of the child body. Labels: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, the ticket is assigned to the driving dev.
- **Blocking**: see **Blocking** under Conventions.
- **Frontier query**: list the map's open children (`gh issue list --state open`, scoped to the map's sub-issues / task list), drop any with an open blocker (see **Count open blockers**, or an open issue in the `Blocked by` line) or an assignee; first in map order wins.
- **Claim**: `gh issue edit <n> --add-assignee @me`, the session's first write.
- **Resolve**: `gh issue comment <n> --body "<answer>"`, then `gh issue close <n>`, then append a context pointer (gist + link) to the map's Decisions-so-far.
