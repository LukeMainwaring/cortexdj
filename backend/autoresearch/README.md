# Autoresearch

A [Karpathy-style](https://github.com/karpathy/autoresearch) autoresearch
loop for the cortexdj EEG classifier: an AI coding agent modifies one
file, launches a short training run on Modal, reads the resulting
metric, and decides whether to keep or revert.

Adapted to EEGNet-on-DEAP, running via Modal (no local GPU).

## Layout

```
backend/autoresearch/
  prepare.py              FROZEN — fixed subject split + evaluate()
  train.py                AGENT EDITS — model, optimizer, loop
  program.md              agent instructions (read this to understand the loop)
  experiments/            append-only history (gitignored)
    experiments.jsonl     one line per run
    best.json             current champion
    runs/<id>/            per-run: train.py snapshot, stdout.log, metrics.json

backend/scripts/
  modal_autoresearch.py   Modal entry — same shape as modal_train.py
  run_autoresearch.py     thin wrapper the agent invokes
```

## Run one experiment manually

From the repo root:

```bash
uv run --directory backend python scripts/run_autoresearch.py
# equivalent:
modal run backend/scripts/modal_autoresearch.py
```

~30 minutes end-to-end: ~15 min Modal cold-start, 15 min training.
Overnight (8h) you can expect ~12–16 experiments. When each one
finishes, a new line lands in `experiments/experiments.jsonl`; if the
metric improved, `experiments/best.json` updates.

GPU defaults to A10G. Override with `--gpu A100` (or T4/H100).

## Run the agent loop

Start a fresh Claude Code (or Codex) session in this repo:

> Have a look at `backend/autoresearch/program.md` and start running
> experiments. Keep iterating until I come back.

The agent reads the program, forms a hypothesis, edits `train.py`,
invokes the wrapper, reads the JSONL tail, decides keep-or-revert,
and repeats.

## Local dry-run (no GPU, no Modal)

Handy for sanity-checking `train.py` before burning Modal time:

```bash
WALL_CLOCK_BUDGET_SECONDS=30 \
  AUTORESEARCH_RUN_DIR=/tmp/ar_local \
  uv run --directory backend python -m autoresearch.train
```

You'll see `FINAL_METRIC=<float>` and `/tmp/ar_local/metrics.json`.
On CPU with a 30s budget, expect ~0.5 (chance baseline — there isn't
enough time for the model to learn).

## The metric

`avg_macro_f1` — mean of arousal macro-F1 and valence macro-F1,
evaluated on DEAP subjects 29–32. Same definition as production
EEGNet training, but with a **fixed 28/4 subject split** instead of
LOSO — LOSO's 32 folds are too slow for a 15-min wall-clock budget.

Label strategy: `median_per_subject` (matches production default).

## Design choices

- **Single file to edit.** The agent only touches `train.py`. Diffs
  are reviewable; production `src/cortexdj/ml/` stays untouched.
- **Inlined EEGNet.** `train.py` doesn't `import` from
  `cortexdj.ml.model` — the architecture is duplicated inline so the
  agent can freely rewrite it. Production isn't affected.
- **Wall-clock budget.** Training always runs 15 minutes regardless
  of batch size, model depth, or optimizer — keeps runs comparable.
  Override for smoke tests with `WALL_CLOCK_BUDGET_SECONDS=<seconds>`.
- **Stdout marker + JSON.** `FINAL_METRIC=<float>` in stdout plus
  `metrics.json` on disk — belt-and-suspenders so parsing is robust
  whether the run completed cleanly or crashed late.
- **Local-only log.** `experiments/` lives on your laptop. When this
  graduates to scheduled/PR-producing runs, we'll promote it to a
  Modal volume (or a git branch).

## What's NOT wired up yet

- `/schedule` or `/loop` orchestration for overnight runs.
- Auto-PR with a summary of top-K experiments + suggested baseline
  update.
- Warm HF cache / preprocessing-preloaded image to cut cold-start.
- Parallel fan-out across multiple Modal GPUs (SkyPilot-style).
- Contrastive pipeline as a second target.

All handled as separate follow-ups.
