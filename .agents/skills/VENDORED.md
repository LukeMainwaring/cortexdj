# Vendored skills

Skills copied from https://github.com/mattpocock/skills. `scripts/check-vendored-skills.sh` reads
this file and reports upstream changes since the pinned commit, plus upstream skills not listed
here. The `updating-deps` skill runs it in Phase 4. Only the `engineering` and `productivity`
buckets are tracked.

Upstream commit: 5863806

## Vendored

Format: `<bucket>/<name>`. A `(customized)` suffix means the local copy carries intentional edits
(review-changes in place of code-review, REVIEW.md in place of CODING_STANDARDS.md, the readiness
gate from docs/agents/issue-tracker.md, the PR rules in AGENTS.md); apply upstream changes to it by hand, never by copying.

- engineering/codebase-design
- engineering/diagnosing-bugs
- engineering/domain-modeling
- engineering/grill-with-docs
- engineering/implement (customized)
- engineering/implement-spec (customized)
- engineering/improve-codebase-architecture
- engineering/pr
- engineering/prototype
- engineering/research
- engineering/retro (customized)
- engineering/setup-matt-pocock-skills (customized)
- engineering/tdd (customized)
- engineering/to-spec
- engineering/to-tickets
- engineering/wayfinder
- productivity/grill-me
- productivity/grilling
- productivity/handoff
- productivity/writing-for-agents

## Not vendored

Upstream skills looked at and skipped. The check does not report these as new.

- engineering/ask-matt
- engineering/code-review
- engineering/triage
- engineering/wizard
- productivity/teach
- productivity/to-questionnaire
- productivity/wait-what
