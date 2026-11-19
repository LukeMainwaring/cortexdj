# Eval baseline

An append-only run log: each section is a dated run, kept as it was recorded. A run is a
measurement to compare later runs against, not a target. Record a run here when it is a
baseline (`EVAL_REPEAT=5` or more, before and after a deliberate prompt or model change) or
when it changes the judge; see `DEVELOPMENT.md`, "Evals".

No run has been recorded yet. The first section follows this shape:

```md
## <What this run measures>

Added <YYYY-MM-DD>, at `<short sha>`. <One or two sentences: what changed since the last
section, or why this is a new baseline rather than a comparison.>

| | |
|---|---|
| Agent model | `<provider:model>` (`AGENT_MODEL`), `AGENT_REASONING_EFFORT` <value or unset> |
| Judge model | `<provider:model>` (`EVAL_JUDGE_MODEL`) |
| Repeat | <n> runs per case (golden set: 1) |
| Golden set | <n> of <n> |
| Case errors | none |
| Cost | about $<n> for the <n> agent runs; judge calls not included |

### <Dataset name>

| Evaluator | Kind | Result |
|---|---|---|
| `<name>` | judge | <passed>/<runs> |
| `<name>` | deterministic | <passed>/<runs> |

### Hand-read notes

- **`<evaluator>`, `<case>` run <n>.** <Real, or the judge being stricter than a person?
  One or two sentences.>

### Changed alongside

- <A change to the eval code made for this run, if any.>

### Left alone

- <Noted and not fixed: the rule for a baseline run is fix one thing, note the rest.>
```
