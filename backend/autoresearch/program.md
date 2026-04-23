# Autoresearch program

You are a research agent. Your job: iteratively improve the validation
macro-F1 of the EEG emotion classifier by modifying ONE file, running
a short experiment on Modal, and logging the result. When you start,
you keep running experiments until instructed to stop.

## The metric

`avg_macro_f1` — mean of arousal macro-F1 and valence macro-F1 on a
fixed held-out set (DEAP subjects 29, 30, 31, 32). Higher is better.
Scale is 0–1; a constant predictor scores 0.5. Current champion is in
`experiments/best.json`.

## Files you may / may not touch

- `train.py` — **you MAY edit.** Model, optimizer, loss, loop. Free
  rein as long as the four contracts below hold.
- `prepare.py` — **you may NOT edit.** Provides the fixed data split
  and evaluation function so metrics stay comparable across runs.
- `program.md` — **you may NOT edit.** The human iterates on this.
- `experiments/` — read it before every experiment.
- Anything outside `backend/autoresearch/` — **off-limits.**

## Contracts (break any and the run is logged as failed)

1. Load data with `load_splits()` from `prepare`.
2. Compute metrics with `evaluate(model, val_ds, device)` from `prepare`.
3. Stop training when `prepare.WALL_CLOCK_BUDGET_SECONDS` has elapsed.
4. At the end, print `FINAL_METRIC=<float>` and write `metrics.json`
   (with at least `avg_macro_f1`) to `$AUTORESEARCH_RUN_DIR`.

## The loop

Run from repo root:

1. Read `backend/autoresearch/experiments/best.json` and the tail of
   `experiments/experiments.jsonl` (last ~10 entries).
2. Form a one-sentence hypothesis. Example: *"A wider FC backbone
   should help given the 160-dim input is already well-separated."*
3. Edit `backend/autoresearch/train.py`.
4. Launch the experiment:
   ```
   uv run --directory backend python scripts/run_autoresearch.py
   ```
   Budget ~30 min per experiment (~15 min Modal cold-start + 15 min
   training). Overnight (8h) you get ~12–16 experiments.
5. Read the last line of `experiments/experiments.jsonl`. If
   `is_best: true`, keep the edit. Otherwise revert:
   ```
   git checkout backend/autoresearch/train.py
   ```
6. Go to 1.

## Idea bank (non-exhaustive)

- **Architecture:** hidden_dim, backbone depth, spatial/temporal filter
  counts, dropout, activation (ELU / GELU / ReLU), skip connections,
  BatchNorm vs LayerNorm, SE blocks, shared vs split heads.
- **Optimizer:** LR schedule (cosine, one-cycle, warmup+decay), AdamW
  betas/eps, weight_decay, SGD+momentum, Muon (see
  karpathy/autoresearch for a minimal PyTorch port).
- **Loss:** label_smoothing, focal loss, per-head loss weighting,
  class-balanced CE, supervised contrastive regularizer.
- **Augmentation:** feature-level Gaussian noise, channel dropout,
  mixup, band masking, input normalization/scaling.
- **Regularization:** weight decay, dropout schedule, stochastic
  depth, gradient clipping, EMA weights.
- **Training dynamics:** batch size, gradient accumulation, mixed
  precision, larger eval frequency for faster best-tracking.

## Guardrails

- If `train.py` crashes or the loss goes NaN/inf, revert immediately.
  The run will land as `status: "failed"` in the log.
- If the metric drops >10% below the baseline in `best.json`, pause
  and flag — something is fundamentally wrong, not just a bad idea.
- Don't commit or push. The log is local.
- Never touch files outside `backend/autoresearch/train.py`.

## What "better" means

Strict improvement on the `metric` column vs `best.json`. Ties don't
count — revert and try something else.
