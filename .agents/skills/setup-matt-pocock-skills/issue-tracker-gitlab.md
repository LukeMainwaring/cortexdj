# Issue tracker: GitLab

Issues and specs for this repo live as GitLab issues. Use the [`glab`](https://gitlab.com/gitlab-org/cli) CLI for all operations.

## Conventions

- **Create an issue**: `glab issue create --title "..." --description "..."`. Use a heredoc for multi-line descriptions. Pass `--description -` to open an editor.
- **Read an issue**: `glab issue view <number> --comments`. Use `-F json` for machine-readable output.
- **List issues**: `glab issue list -O json` with appropriate `--label` filters.
- **Comment on an issue**: `glab issue note <number> --message "..."`. GitLab calls comments "notes".
- **Apply / remove labels**: `glab issue update <number> --label "..."` / `--unlabel "..."`. Multiple labels can be comma-separated or by repeating the flag.
- **Close**: `glab issue close <number>`. `glab issue close` does not accept a closing comment, so post the explanation first with `glab issue note <number> --message "..."`, then close.
- **Merge requests**: GitLab calls PRs "merge requests". Use `glab mr create`, `glab mr view`, `glab mr note`, etc., the same shape as `gh pr ...` with `mr` in place of `pr` and `note`/`--message` in place of `comment`/`--body`.

Infer the repo from `git remote -v`; `glab` does this automatically when run inside a clone.

## Merge requests as a triage surface

**MRs as a request surface: no.** _(Set to `yes` if this repo treats external merge requests as feature requests; `/triage` reads this flag.)_

When set to `yes`, MRs run through the same labels and states as issues, using the `glab mr` equivalents:

- **Read an MR**: `glab mr view <number> --comments` and `glab mr diff <number>` for the diff.
- **List external MRs for triage**: `glab mr list -O json`, then keep only MRs whose author is not a project member/owner (a contributor's MR, not a maintainer's in-flight work).
- **Comment / label / close**: `glab mr note`, `glab mr update --label`/`--unlabel`, `glab mr close`.

Unlike GitHub, GitLab numbers issues and MRs separately, so `#42` is unambiguous once you know which surface the maintainer means.

## Status

Specs and implementation tickets carry one of these labels:

- `ready-for-agent`: eligible once its hard blockers are resolved (see Implementation readiness)
- `needs-planning`: waits for a planning session

A ticket is resolved when it is closed. Close it with a comment naming the MR that completed it (`MR #N`); a spec finished across several MRs lists them. A spec resolves with its last ticket, or on its own when implemented directly.

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

This tracker closes work through merge requests.

1. **Resolve on approval.** After review, wait for the user's explicit approval to merge. Then, before merging, close each ticket the MR completes (and its spec, if this finishes it) with a comment naming the MR, so the MR carries its own bookkeeping.
2. **Merge.** If the merge fails (conflicts, failing checks, permissions), report the failure and its error plainly. A closed ticket records reviewed, approved work; it does not mean the merge happened, so never say it did.
3. **Clean up after a successful merge.** First check there is nothing left to lose: `git fetch`, then `git merge-base --is-ancestor <branch> origin/main` must succeed, and the worktree must have no uncommitted changes. (`git branch -d` alone is not that check: it compares against the branch's upstream, not `main`.) Then remove the worktree, with the environment's worktree tool when the session created it, otherwise `git worktree remove`; then delete the branch locally (`git branch -d`) and on the remote. Leave everything in place when the check fails, or the branch is a research or prototype branch or unrelated to this work. Review branches stay until their merge succeeds.

## When a skill says "publish to the issue tracker"

Create a GitLab issue.

## When a skill says "fetch the relevant ticket"

Run `glab issue view <number> --comments`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body. `glab issue create --label wayfinder:map`. (On GitLab tiers with native epics, an epic may hold the map instead; a labelled issue works everywhere.)
- **Child ticket**: an issue carrying `Part of #<map>` at the top of its description, linked to the map (`glab issue create --linked-issues <map> --link-type relates_to`), and labels `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, the ticket is assigned to the driving dev.
- **Blocking**: GitLab's **native blocking link**, the canonical, UI-visible representation. Add it with the `/blocked_by #<n>` quick action, posted as a note (`glab issue note <child> --message "/blocked_by #<blocker>"`). Native blocking links are a Premium/Ultimate feature; on the free tier (or where unavailable) fall back to a `Blocked by: #<n>, #<n>` line at the top of the description. A ticket is unblocked when every blocker is closed.
- **Frontier query**: list the map's children with `glab api projects/:id/issues/<map>/links`, keeping open issues that carry a `wayfinder:<type>` label; drop any with an open blocker: a native `blocked_by` link to an open issue (`glab api projects/:id/issues/<child-iid>/links`), or an open issue in the `Blocked by` line, or an assignee; first in map order wins.
- **Claim**: `glab issue update <n> --assignee @me`, the session's first write.
- **Resolve**: `glab issue note <n> --message "<answer>"`, then `glab issue close <n>`, then append a context pointer (gist + link) to the map's Decisions-so-far.
