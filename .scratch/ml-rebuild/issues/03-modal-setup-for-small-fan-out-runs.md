# 03: Modal's current setup for small fan-out training runs

**Type:** research
**Status:** resolved

## Question

What does Modal recommend today for a small PyTorch training job that fans out about 32 independent folds: how the image is built from a uv project, how a licensed dataset of about 3GB gets onto a volume once and is read by every fold, how folds fan out (`.map` or `.starmap`) and return metrics and a checkpoint, whether a GPU is worth it against CPU for a model that trains in minutes, cold-start time, and what a full 32-fold run would cost at current prices. Also: the thinnest way to wrap a plain Python training function so the same function runs locally unchanged. The old wrapper in git history (`legacy/backend/scripts/modal_train.py`, removed in 2378eff) shows what was done before.

## Answer

Full findings: `docs/research/modal-fan-out-training.md` on branch `refs/heads/research/ml-rebuild-modal-fan-out` (commit 973b485, pushed, never merged). Read 2026-11-22 from Modal's docs, SDK release notes (1.6.4) and pricing page. Nothing was run on Modal, so every sketch is untested.

- **Image:** `modal.Image.debian_slim(python_version="3.14").uv_sync("backend", extra_options="--no-dev").add_local_python_source("cortexdj")`. `uv_sync` copies only `pyproject.toml` and `uv.lock`, so a code edit does not rebuild the dependency layer. The old wrapper copied `backend/` before syncing, the likely cause of its 15-minute cold starts (not reproduced).
- **Data:** create a Volume once, `modal volume put` the DEAP folder from the laptop, mount it read-only in the fold function. Storage for 3GB is free.
- **Fan-out:** `fold.map(range(1, 33))` from a `@app.local_entrypoint()`. Each fold returns a dict of metrics plus the checkpoint as bytes, and the entrypoint writes them to local disk; no Volume download. Set `timeout` (default is 300 seconds) and `retries`; `return_exceptions=True` keeps the good folds if one fails.
- **CPU, not GPU:** at an assumed 5 minutes a fold, a 32-fold run costs about $0.67 on CPU (4 cores, 8GiB) against about $2.00 on a T4, and CPU runs all 32 folds at once where the Starter plan's 10-GPU limit forces waves. "GPU not worth it" is the researcher's judgment from the model's size, not a Modal statement. Real fold time, memory, and cold start on Modal are unmeasured.
- **Wrapper:** one Modal file of about 30 lines. An `@app.function` imports `train_fold` inside its body and calls it with the Volume's data path; locally the same function is called with the local path. The training code has no Modal import, and `modal` can live in the `dev` group. A single `--gpu` option via `Function.with_options` replaces the old `@app.cls` workaround.
- **Goes away from the old wrapper:** `.modalignore`, the `uv run` subprocess with passthrough args, fold-level resume state, the 4-hour timeout, `caffeinate`, the `A10G` default.
- **Not verified:** `uv_sync` against this lockfile on Python 3.14; whether Volumes v2 is still marked experimental in the CLI; the size limit on a returned payload; whether the old `cortexdj-deap` Volume still exists; DEAP's licence terms on keeping a copy with a cloud provider.
