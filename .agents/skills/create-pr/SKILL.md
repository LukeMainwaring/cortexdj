---
name: create-pr
description: Summarize the current branch's changes and open a pull request with gh.
allowed-tools: Bash(git:*) Bash(gh:*)
disable-model-invocation: true
---

# PR Summary

Open a pull request for the current branch, targeting the repository's default
branch.

## Instructions

1. **Determine the base branch** (the repo's default branch — `main` here). The
   PR always targets this branch; stacked PRs onto a parent feature branch are
   not supported.

    ```bash
    base="$(gh repo view --json defaultBranchRef -q .defaultBranchRef.name 2>/dev/null || echo main)"
    ```

2. **If you are on the base branch, create a feature branch first** (a PR cannot
   go from the base branch to itself):

    ```bash
    [ "$(git branch --show-current)" = "$base" ] && git checkout -b <descriptive-branch-name>
    ```

3. **If there are uncommitted changes (staged or unstaged), commit them** with a
   clear message describing what changed. Do not add `Co-Authored-By` or
   "Generated with Claude Code" lines by hand — Claude Code adds attribution
   natively via the `attribution` setting, and hand-writing it fights a user who
   has turned it off.

4. **Analyze the changes** against the base (prefer the pushed ref `origin/$base`,
   fall back to the local `$base` if it isn't fetched):

    ```bash
    ref="origin/$base"; git rev-parse --verify --quiet "$ref" >/dev/null || ref="$base"
    git log "$ref..HEAD" --oneline
    git diff "$ref...HEAD" --stat
    ```

5. **Generate a summary** with:

    - Brief description of what changed
    - List of files modified
    - Breaking changes (if any)
    - Testing notes

6. **Format as the PR body**:

    ```markdown
    ## Summary

    [1-3 bullet points describing the changes]

    ## Changes

    -   [List of significant changes]

    ## Test Plan

    -   [ ] [Testing checklist items]
    ```

7. **Push the branch** (set upstream on first push):

    ```bash
    git push -u origin HEAD
    ```

8. **Open the pull request**, targeting the detected base so the PR target matches
   the diff above:

    ```bash
    gh pr create --base "$base" --title "<title>" --body "<the markdown from step 6>"
    ```

9. **Return the PR URL when done.**
