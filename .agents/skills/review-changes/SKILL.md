---
name: review-changes
description: "Review this branch's changes against main with the harness's built-in code review plus REVIEW.md, with an optional second opinion from the other CLI (Claude Code ⇄ Codex). Use only when the user or an invoking skill explicitly asks to review changes, a branch, or a PR."
---

# Review Changes

A thin wrapper. The built-in reviewer (`/code-review` in Claude Code, `codex review` in Codex)
finds and verifies; `REVIEW.md` at the repo root carries the repo-specific review checks, and
AGENTS.md sends every reviewer there; this skill only picks the mode, runs the passes, and
reconciles. Do not review the diff yourself in place
of the built-in, and do not restate `REVIEW.md` in the prompts you build.

## Arguments

All optional, in any order:

- **mode**: `solo` (one built-in pass) or `cross-model` (also runs the other CLI). If absent,
  ask the user once: "Solo, or cross-model with a second opinion from <other CLI>?" A skill
  that invokes this one unattended must pass a mode; never ask when a mode was given.
- **target**: what to review. Default is the branch diff, `main...HEAD`. A PR number, branch,
  ref range, or path is accepted. Resolve it to commit ids once (merge base → head, plus any
  path filter) and give every pass the same endpoints, so uncommitted edits stay out unless
  the target explicitly includes them.
- **spec**: a path or issue reference for the change's originating spec. Enables the Spec pass.
- **effort** (Claude Code only): a `/code-review` level (`low` … `max`). Passed through as
  given; otherwise the built-in's own default applies.

## Process

1. **Resolve the mode** (see above). Then check the other CLI exists (`command -v codex` or
   `command -v claude`). If `cross-model` was chosen and it is missing or unauthenticated,
   continue in `solo` and say so in the Summary. Never claim a second opinion that did not run.

2. **Primary pass**, using your own harness's built-in:
   - **Claude Code**: invoke the built-in `code-review` skill with the resolved target passed
     explicitly (`main...HEAD` by default; its own default would add uncommitted changes), and
     the effort if given.
   - **Codex**: run the Codex review command below, with `primary.md` as the output file.

3. **Second opinion** (`cross-model` only), running the *other* CLI's built-in on the same target:
   - **From Claude Code**: run the Codex review command below, with `second.md` as the output file.
   - **From Codex**: `out=$(mktemp -d) && claude -p "/code-review <target>" > "$out/second.md" && echo "$out"`.

   Neither built-in spawns the other, so there is nothing to guard against recursion.

   **The Codex review command.** Run it as one shell invocation exactly as written, since
   shell variables do not survive between tool calls. It creates its own output directory,
   computes the merge base, and prints the directory path so you can read the result file:

   ```bash
   out=$(mktemp -d) && head=$(git rev-parse HEAD) && base=$(git merge-base main "$head") && codex exec review --ephemeral "Review the code changes against the base branch 'main'. The merge base commit is $base and the head commit is $head. Review only \`git diff $base $head\`." > "$out/<file>" && echo "$out"
   ```

   Replace `<file>` with `primary.md` or `second.md`. For a non-default target, substitute the
   resolved base and head commits (and add the path filter, if any) in the prompt. If the
   command exits non-zero or the file is empty, the pass did not run: say so, never summarize
   an empty file as "no findings".

4. **Spec pass** (only when `spec` was given). Read the spec. Report three things, quoting the
   spec line for each: requirements that are missing or partial; behaviour in the diff nobody
   asked for; requirements that look implemented but wrong. Under 400 words. In Claude Code,
   run this in a subagent so it does not share context with the code passes.

5. **Reconcile** (`cross-model` only). Findings both passes raised are deduplicated into one,
   high confidence, source `both`. A finding only the other pass raised is re-checked against
   the actual code and kept only if it holds, source tagged. A finding only your pass raised is
   kept, source tagged. Settle disagreements by re-reading the code, not by trusting the report
   you find more familiar. The reconciled set is what gets reported.

6. **Report**:
   - **Summary**: one line on change quality, then which passes actually ran.
   - **Findings**: in `solo` mode, the primary pass's findings exactly as the built-in produced
     them. In `cross-model` mode, the reconciled union from step 5, deduplicated and ordered by
     severity, in the primary reviewer's finding format, each ending with
     `(source: claude | codex | both)`.
   - **Spec** (only if run): the Spec pass report under its own heading, kept separate from the
     code findings.
   - **Verdict**: the line `REVIEW.md` asks for. If the primary pass did not end with one, add it
     from the findings: **Ship it** (nothing above nit), **Fix and ship** (minor items), or
     **Needs changes** (a correctness bug or a real convention or architecture breach).

Findings are advisory. Never apply fixes from this skill. The user can run the built-in with
its own fix flag if they want that.
