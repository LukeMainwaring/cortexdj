# 0004: Eval judges run on a model a tier above the agents, pinned apart from them

**Status:** accepted (2026-10-17)

## Context

The rubric judges in `tests/evals/judges.py` grade every agent reply the deterministic checks
cannot: whether a greeting is friendly, whether a reply names one next step, whether a figure was
invented. A judge that is the same model as the agent it grades shares its habits: a style of
reply it likes in itself, or an error it cannot see, passes unnoticed. A judge weaker than the
agent misses what the agent misses. And a judge that follows `AGENT_MODEL` changes whenever the
agent's model does, so a before-and-after on a model swap compares two different graders.

## Decision

- **`EVAL_JUDGE_MODEL` is pinned and never follows `AGENT_MODEL`.** Its default is a tier above
  the agents' default on the same provider (Opus 5.5 over Sonnet 5.5, with Haiku 5.5 for titles).
  Changing an agent's model never changes who grades it.
- **The golden set is the gate on any change to the judge or a rubric.** `tests/evals/golden_set.py`
  is the hand-labeled calibration: one good and one bad reply per judged behavior, and a judge
  that misgrades any of them is not used. A rubric with no golden pair is not calibrated.
- **The pairing is a convention, not a check.** Nothing stops an operator setting `AGENT_MODEL`
  to the judge's model and grading it with itself. Each run recorded in `docs/evals/baseline.md`
  names the agent and judge models it ran with, so a reader can see the pairing; a code check
  that compares model tiers would need a tier table that goes stale with every release.

## Considered options

| Option | Why not |
| --- | --- |
| Judge on `AGENT_MODEL` | The judge grades its own habits, and swapping the agent's model swaps the grader too. |
| The cheapest model as the judge | A judge weaker than the agent misses what the agent misses; a smaller model fails the arithmetic a grounding rubric asks for. The cost-down experiment to try later, gated on the golden set. |
| A model from another family | Removes the shared-family blind spot entirely, which a tier up only narrows. It spends the other provider's budget on every eval run and needs that provider's key for a run that otherwise needs one. Worth revisiting as a second judge for close calls. |

## Consequences

- Judge spend is about a cent a call, a small share of a run, which the agents dominate.
- The eval tier spends on one provider, the judge's, unless an agent runs on another.
- Code comments and docs point here for the rationale rather than restating it.
