---
name: create-pr
description: Generate a summary for the current branch changes
allowed-tools: Bash(git:*, gh:*)
disable-model-invocation: true
---

# PR Summary

Generate a pull request summary for the current branch.

## Instructions

1. **If any untracked changes, commit with a clear message based on what changed**

2. **Analyze changes**:

    ```bash
    git log main..HEAD --oneline
    git diff main...HEAD --stat
    ```

3. **Generate summary** with:

    - Brief description of what changed
    - List of files modified
    - Breaking changes (if any)
    - Testing notes

4. **Format as PR body**:

    ```markdown
    ## Summary

    [1-3 bullet points describing the changes]

    ## Changes

    -   [List of significant changes]

    ## Test Plan

    -   [ ] [Testing checklist items]
    ```

5. **Use `gh pr create` to open a pull request with the above markdown**

6. **Return the PR URL when done**
