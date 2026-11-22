# Modal for a 32-fold PyTorch fan-out, November 2026

Researched 2026-11-22, against primary sources only: Modal's docs fetched as Markdown (`https://modal.com/docs/<page>.md`), the Python SDK release notes (latest release 1.6.4, 2026-11-11), the pricing page, the SDK source for `Image.uv_sync` (`modal-labs/modal-client`, `py/modal/_image.py` on `main`), and Modal's own hyperparameter-sweep and long-training examples. The old wrapper was read from git history (`git show 2378eff^:legacy/backend/scripts/modal_train.py`, `modal_autoresearch.py`, `.modalignore`).

**Confidence note:** nothing was run on Modal and no account or token was touched, so every code sketch below is assembled from the docs and is untested. Timings for our own model on Modal hardware are estimates, not measurements. The list at the end says what could not be verified.

## Summary

1. **Image:** `modal.Image.debian_slim(python_version="3.14").uv_sync("backend", extra_options="--no-dev")`, then `.add_local_python_source("cortexdj")`. `uv_sync` copies only `pyproject.toml` and `uv.lock`, so a code change no longer rebuilds the dependency layer. That rebuild is the most likely cause of the old wrapper's slow starts.
2. **Data:** create a Volume once, upload the 32 files once from the laptop with `modal volume put`, mount it read-only in the fold function. Storage is free at this size.
3. **Fan-out:** `Function.map` over the 32 subject ids from a `@app.local_entrypoint()`. Each call returns a small dict of metrics plus the checkpoint as bytes; the entrypoint writes them to local disk. No Volume download step is needed for a model this small.
4. **GPU:** not worth it for this model. A CPU run of all 32 folds costs about $0.70 at 5 minutes a fold; a T4 run costs about $2.00 at the same duration. Both are far inside the $30 monthly credit. On the Starter plan CPU also fans out wider (100 containers against 10 GPUs).
5. **Wrapper:** one Modal file of about 30 lines whose function body imports the plain training function and calls it. The training code has no Modal import, and locally it is called directly.

## 1. Building the image from the uv project

Modal has had first-class uv support since SDK 1.1.0 (2025-07-17): `Image.uv_pip_install` and `Image.uv_sync` ([release notes, 1.1.0](https://modal.com/docs/reference/changelog)). The images guide now uses `uv_pip_install` as the default way to add packages ([Images guide](https://modal.com/docs/guide/images)).

`uv_sync` signature, from the [Image reference](https://modal.com/docs/sdk/py/latest/Image#uv_sync):

```python
uv_sync(self, uv_project_dir="./", *, force_build=False, groups=None,
    extras=None, frozen=True, extra_options="", uv_version=None, env=None,
    secrets=None, gpu=None)
```

What the reference and the source say it does:

- It adds `pyproject.toml` and `uv.lock` from `uv_project_dir` to the build context. The path is relative to the directory `modal` is called from, so from the repo root it is `uv_sync("backend")`.
- `frozen=True` is the default, so the build runs `uv sync --frozen` against our lockfile.
- It does **not** install the project itself (the source passes `--no-install-workspace`). The reference says to add local source afterwards with `Image.add_local_python_source`.
- The source runs `uv sync --project=/.uv --no-install-workspace --compile-bytecode` and then sets `ENV PATH=/.uv/.venv/bin:$PATH` ([`_image.py`](https://github.com/modal-labs/modal-client/blob/main/py/modal/_image.py)).
- uv workspaces are not supported. `backend/pyproject.toml` is not a workspace, so this does not affect us.
- It passes no `--no-dev`, and uv syncs the `dev` group by default ([uv docs, dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/)). Pass `extra_options="--no-dev"` to keep mypy, ruff and pytest out of the image.

Python 3.14 is supported since SDK 1.3.0 (2025-12-19) ([release notes](https://modal.com/docs/reference/changelog)). `debian_slim` defaults to the local interpreter's minor version ([Images guide](https://modal.com/docs/guide/images)); pass `python_version="3.14"` so the base matches `requires-python = ">=3.14,<3.15"` regardless of where `modal` is launched from.

Local code: `add_local_python_source("cortexdj")` adds the package to `/root`, which is on `PYTHONPATH`, and by default adds it at container start, not as an image layer, so a code edit triggers no rebuild ([Image reference](https://modal.com/docs/sdk/py/latest/Image#add_local_python_source)). It includes only `.py` files by default. Since Modal 1.0 local packages are no longer auto-mounted; only the module that defines the App is included automatically ([1.0 migration guide](https://modal.com/docs/guide/modal-1-0-migration), [Project structure](https://modal.com/docs/guide/project-structure)).

CUDA: the driver and CUDA Driver API are preinstalled on GPU machines, and `torch` from PyPI bundles the rest ([CUDA guide](https://modal.com/docs/guide/cuda)). Our `uv.lock` already resolves the `nvidia-*` wheels for Linux x86_64, so the same image runs on CPU or GPU.

Sketch:

```python
image = (
    modal.Image.debian_slim(python_version="3.14")
    .uv_sync("backend", extra_options="--no-dev")
    .add_local_python_source("cortexdj")
)
```

The image installs the whole backend dependency set (FastAPI, Pydantic AI, transformers), not only what training needs. That costs build time once and nothing afterwards, because layers are cached until the definition changes ([Images guide, caching](https://modal.com/docs/guide/images)). A separate training dependency group would shrink it, at the price of a second setting; not recommended for a first version.

## 2. Getting the DEAP files onto a Volume once

Volumes are Modal's distributed file system, "optimized for write-once, read-many I/O workloads" ([Volumes guide](https://modal.com/docs/guide/volumes)). That is exactly this case: 32 files written once, read by every fold.

One-time setup from the laptop ([`modal volume` CLI reference](https://modal.com/docs/cli/latest/volume)):

```bash
modal volume create --version=2 cortexdj-deap
modal volume put cortexdj-deap backend/data/deap /deap
modal volume ls cortexdj-deap /deap
```

- The guide recommends Volumes v2 for all new Volumes and notes that v1 is still the default when no version is given. The CLI reference still labels `--version` "(Experimental)". For 32 read-only files either version works; v1's limits (best under 50,000 files, about five concurrent writers) do not bind here.
- The same upload is available from Python with `Volume.batch_upload()` ([Volume reference](https://modal.com/docs/sdk/py/latest/Volume#batch_upload)), if a seed script is preferred over a CLI line.
- The dataset-ingestion guide's advice (download inside a Function, large ephemeral disk) is for datasets over 100 GiB or hundreds of thousands of files ([Large dataset ingestion](https://modal.com/docs/guide/dataset-ingestion)). It does not apply to 3 GB.

Reading from every fold:

```python
deap = modal.Volume.from_name("cortexdj-deap")

@app.function(volumes={"/data/deap": deap.with_mount_options(read_only=True)})
```

`with_mount_options(read_only=True)` was added in SDK 1.4.3 (2026-05-18) and applies per mount, not to the Volume ([Volume reference](https://modal.com/docs/sdk/py/latest/Volume#with_mount_options)). A container mounts the Volume's latest state when it starts, so there is no `reload()` to call. Leaving out `create_if_missing` makes a missing Volume fail fast, which the old wrapper also did on purpose.

Licence: the data is not baked into the image and is not passed as function arguments; it sits in a Volume private to the workspace. Whether DEAP's licence permits storing a copy with a cloud provider is a question for the licence text, which this research did not read.

Cost: Volumes are $0.09 per GiB per month with 1 TiB per month free ([pricing](https://modal.com/pricing), read 2026-11-22), so 3 GB costs nothing.

## 3. Fanning out the folds and getting results back

The scaling guide's answer for "the same function repeatedly with different independent inputs (e.g., a grid search)" is `Function.map` ([Scaling out](https://modal.com/docs/guide/scale)). Signatures from the [Function reference](https://modal.com/docs/sdk/py/latest/Function):

```python
map(self, *input_iterators, kwargs={}, order_outputs=True,
    return_exceptions=False, wrap_returned_exceptions=None)
starmap(self, input_iterator, *, kwargs={}, order_outputs=True,
    return_exceptions=False, wrap_returned_exceptions=None)
spawn_map(self, *input_iterators, kwargs={})
```

Which one:

| Method | Use here? | Why |
|---|---|---|
| `.map(range(1, 33))` | Yes | One argument per fold; results come back to the caller as an iterator, in input order by default. `kwargs=` passes the same settings to every fold. |
| `.starmap([...])` | Only if a fold takes several varying positional arguments | Same semantics as `map`, each item unpacked. Modal's own hyperparameter sweep uses it with `order_outputs=False` ([example](https://modal.com/docs/examples/hp_sweep_gpt)). |
| `.spawn_map(...)` | No | Fire-and-forget. It returns no handle, so it only suits functions that write results to storage ([invocation methods](https://modal.com/docs/guide/function-invocation-methods)). |

Behaviours that matter for a 32-fold run:

- `map` is synchronous. If the laptop disconnects, the calls are cancelled within two minutes, and an ephemeral App stops when the client exits unless run with `modal run --detach` ([invocation methods](https://modal.com/docs/guide/function-invocation-methods), [Apps guide](https://modal.com/docs/guide/apps)). For a run of a few minutes, staying attached is fine.
- A raised exception in any fold propagates by default. `return_exceptions=True` puts it in the result stream instead, so 31 good folds are not lost to one bad one ([Scaling out](https://modal.com/docs/guide/scale#exceptions)).
- Each input is retried independently when `retries=` is set ([Failures and retries](https://modal.com/docs/guide/retries)). Functions are preemptible by default and a preempted input is restarted from the beginning ([Preemption](https://modal.com/docs/guide/preemption)). With folds of a few minutes, `retries=2` is all the resilience needed; the old fold-level resume state is unnecessary once a fold is its own input.
- The default timeout is 300 seconds per attempt ([Timeouts](https://modal.com/docs/guide/timeouts)). A fold that takes "a few minutes" is close to that; set `timeout` explicitly.

Returning results. Arguments and return values are serialized with cloudpickle; payloads up to 2 MiB go inline and larger ones through object storage ([Passing local data](https://modal.com/docs/guide/local-data)). The images guide warns to be careful returning objects whose package is missing locally ([Images guide](https://modal.com/docs/guide/images)). So return plain Python: a dict of floats, plus the checkpoint as `bytes` (`torch.save` into a `BytesIO`). The local entrypoint then writes the metrics and the checkpoint files under `backend/data/`.

The "final checkpoint" in a leave-one-subject-out run is normally a model trained on all 32 subjects, separate from the 32 fold models. That is one more input: either an extra `.remote()` call after the map, or a 33rd item in the map with no held-out subject.

When a Volume is the better return path: if checkpoints grow to hundreds of megabytes, or the run should survive the laptop closing. Then each fold writes to a second Volume (commits happen in the background and at container exit), the run uses `spawn_map` with `--detach`, and the files come back with `modal volume get <volume> <remote> <local>` ([Volumes guide](https://modal.com/docs/guide/volumes), [CLI reference](https://modal.com/docs/cli/latest/volume)). Not needed for an EEGNet-sized model.

## 4. GPU or CPU, cold start, and cost

### Prices read on 2026-11-22

From [modal.com/pricing](https://modal.com/pricing), per second, with the hourly figure computed from it:

| Resource | Per second | Per hour |
|---|---|---|
| CPU, per physical core (2 vCPU) | $0.0000131 | $0.047 |
| Memory, per GiB | $0.00000222 | $0.008 |
| Nvidia T4 | $0.000164 | $0.59 |
| Nvidia L4 | $0.000222 | $0.80 |
| Nvidia A10 | $0.000306 | $1.10 |
| Nvidia L40S | $0.000542 | $1.95 |
| Volumes | $0.09 per GiB per month, 1 TiB free | |

Billing rules from the same page and the [resources guide](https://modal.com/docs/guide/resources):

- GPU time is billed on top of CPU and memory; the pricing FAQ's GPU example lists all three charges.
- CPU and memory are billed at the higher of the request and actual use. The default request is 0.125 cores and 128 MiB.
- Billable time includes application load and, by default, 60 idle seconds per container after its last input (`scaledown_window`, [Cold start guide](https://modal.com/docs/guide/cold-start)).
- Starter plan: $0 per month, $30 of compute credit per month, 100 containers and 10 GPUs concurrently.

### Cost of a 32-fold run

These are estimates. The inputs that are assumptions, not facts: a fold takes 5 minutes on Modal (the owner's "a few minutes" on a laptop), a CPU fold requests 4 cores and 8 GiB, a GPU fold requests 2 cores and 8 GiB. Idle and startup seconds are left out; at 60 idle seconds per container they add under 25%.

| Setup | Rate per container | 32 folds at 5 min | 32 folds at 10 min |
|---|---|---|---|
| CPU, 4 cores + 8 GiB | $0.25/h | $0.67 | $1.35 |
| T4 + 2 cores + 8 GiB | $0.75/h | $2.00 | $3.99 |
| A10 + 2 cores + 8 GiB | $1.26/h | $3.36 | $6.72 |

The $30 monthly credit covers about 45 full CPU runs at 5 minutes a fold, or about 15 T4 runs. A budget cap can be set at the workspace level ([Billing guide](https://modal.com/docs/guide/billing)).

### Is a GPU worth it

Modal's docs do not answer this for small models; the GPU guide's advice is to "understand where the bottlenecks are" before reaching for a bigger GPU ([GPU guide](https://modal.com/docs/guide/gpu)). The rest of this subsection is this research's judgment, not a Modal statement.

- The classifier in `backend/src/cortexdj/ml/model.py` is two small convolutions and three linear layers over band-power features. A model that size is usually limited by data loading and Python overhead, not by matrix arithmetic, so a GPU buys little.
- The fan-out is where the time is saved. On CPU, 32 folds run at once and the run takes about as long as one fold. On the Starter plan's 10-GPU limit, 32 GPU folds run in four waves.
- GPU functions cannot be made non-preemptible ([Preemption](https://modal.com/docs/guide/preemption)), and GPUs require a payment method on file ([GPU guide](https://modal.com/docs/guide/gpu)).
- Recommendation: default to CPU, and keep `gpu` as the one optional setting. Since SDK 1.4.3 a plain function can be switched at call time with `fold.with_options(gpu="T4")` ([Function reference](https://modal.com/docs/sdk/py/latest/Function#with_options)), so trying a GPU later is one flag, not a second code path. Measure one fold each way before paying for 32.

### Cold start

What the docs state ([Cold start guide](https://modal.com/docs/guide/cold-start), [pricing FAQ](https://modal.com/pricing)): containers boot in about one second, and the time until a container is ready "can range from seconds to minutes" depending on global-scope imports and `enter` methods. Importing `torch` and reading 3 GB from the Volume happen after boot; the guide gives no number for either. Volumes are "designed to provide up to 2.5 GB/s", not guaranteed.

The first run after a dependency change also builds the image, which for a torch-sized environment takes minutes. Later runs reuse the cached layers. `docs/ROADMAP.md` records a "~15-min cold-start on each run" for the old wrapper; section 6 explains the likely cause.

## 5. The thinnest wrapper

The documented pattern for code that should not import Modal is to import it inside the function body ([Images guide](https://modal.com/docs/guide/images)). Sketch of the whole wrapper, assuming a plain `train_fold(subject, data_dir) -> dict` exists somewhere under `cortexdj` (names illustrative):

```python
# backend/scripts/modal_train.py, run from the repo root with:
#   modal run backend/scripts/modal_train.py
from pathlib import Path

import modal

app = modal.App("cortexdj-train")
image = (
    modal.Image.debian_slim(python_version="3.14")
    .uv_sync("backend", extra_options="--no-dev")
    .add_local_python_source("cortexdj")
)
deap = modal.Volume.from_name("cortexdj-deap")


@app.function(
    image=image,
    volumes={"/data/deap": deap.with_mount_options(read_only=True)},
    cpu=4,
    memory=8192,
    timeout=1800,
    retries=2,
)
def fold(subject: int) -> dict:
    from cortexdj.training import train_fold  # plain function, no Modal import

    return train_fold(subject, data_dir=Path("/data/deap"))


@app.local_entrypoint()
def main(gpu: str = "") -> None:
    run = fold.with_options(gpu=gpu) if gpu else fold
    for result in run.map(range(1, 33), order_outputs=False):
        ...  # write result["metrics"] and result["checkpoint"] under backend/data/
```

Why this shape:

- **One code path.** On the laptop, the single-fold command calls `train_fold(subject, data_dir=local_path)` directly. On Modal, the same function is called with the Volume path. The only thing that differs is `data_dir`.
- **`Function.local()` is not needed.** It exists and "is equivalent to calling the unwrapped underlying function directly" ([invocation methods](https://modal.com/docs/guide/function-invocation-methods)), but using it would make the local path import Modal for no gain.
- **Settings.** `modal run` turns the entrypoint's primitive arguments into CLI options ([Apps guide](https://modal.com/docs/guide/apps)), so `--gpu T4` is the whole settings surface. Anything about the training itself belongs to `train_fold`'s own arguments, forwarded with `map(..., kwargs={...})`.
- **Where `modal` is installed.** It is needed only on the laptop, to launch. On image builder versions after 2024.10 the client is mounted into containers at run time and need not be in `uv.lock` (comment in the `uv_sync` source). Putting `modal` in the `dev` group and syncing the image with `--no-dev` fits that.

Decorating the imported function directly, `app.function(...)(train_fold)`, would remove the three-line body. The docs do not show that form for a function defined in another module, so it is left as unverified below.

## 6. What the old wrapper did, and what would change

The old `legacy/backend/scripts/modal_train.py` (about 170 lines) built from `ghcr.io/astral-sh/uv:python3.14-bookworm-slim`, copied `backend/` into the image, ran `uv sync` as a build command, and ran the whole multi-fold training as one `subprocess.run(["uv", "run", ...])` on one A10G with a 4-hour timeout, returning every `*.pt` as bytes.

| Old wrapper | Current practice | Why |
|---|---|---|
| `from_registry(uv image)` + `add_local_dir(backend, copy=True)` + `run_commands("uv sync ...")` | `debian_slim(python_version="3.14").uv_sync("backend")` + `add_local_python_source("cortexdj")` | The old order put the source copy before the sync, so any source edit invalidated the dependency layer ("breaking the cache on a single layer will cause cascading rebuilds", [Images guide](https://modal.com/docs/guide/images)). This is the likely cause of the recorded 15-minute starts; it was not measured. |
| `add_local_file(README.md)` because hatchling reads `readme = "../README.md"` | Not needed | `uv_sync` does not build or install the project. |
| A 13-line `BACKEND_IGNORE` list and a `.modalignore` | Not needed | Only `pyproject.toml`, `uv.lock` and the package's `.py` files are sent. |
| `@app.cls` used only to reach `Cls.with_options(gpu=...)`, with a comment that functions do not support it | Plain `@app.function` + `Function.with_options` | Added in SDK 1.4.3 (2026-05-18). |
| One container runs all folds in sequence; fold-level resume state on the Volume; `caffeinate` for long runs | One input per fold through `.map`, `retries=2` | Each fold is short and independently retried, so resume state and the 4-hour timeout go away. |
| `subprocess.run(["uv", "run", "train-model", ...])` with a passthrough `--args` string | Import and call the function | The venv is on `PATH`; no subprocess or argument string. |
| Volume mounted read-write, `deap_volume.commit()` for a preprocessing cache | Read-only mount | Thirty-two containers writing the same cache files would hit the last-write-wins rule ([Volumes guide](https://modal.com/docs/guide/volumes)). If a feature cache is wanted, build it once in a separate step. |
| `gpu="A10G"` default | CPU default, `gpu` optional | See section 4. The GPU guide now lists the type as `A10`; Modal's own sweep example still passes `"A10G"`. |
| `apt_install("ffmpeg")` | Dropped for the classifier | It served contrastive training's audio decoding only. |

`modal_autoresearch.py` shared the same image and added a per-run log; its image notes apply unchanged.

## Not verified

- **Nothing was executed on Modal.** The sketches follow documented signatures but have not been built or run, including `uv_sync` against this lockfile on Python 3.14.
- **Fold duration on Modal hardware**, on CPU or GPU, and so the real cost. The cost table rests on an assumed 5 or 10 minutes a fold and assumed CPU and memory requests. Peak memory for a fold that loads 31 subjects was not measured.
- **Cold-start seconds** for this image (torch import, Volume read). The docs give "about one second" for container boot and no figure for the rest. Image build time was not measured.
- **Whether the 15-minute starts came from the layer order.** It follows from the caching rule and the old file, but was not reproduced.
- **`app.function(...)(imported_function)`** for a function defined in another module, without `serialized=True`. Not shown in the docs read.
- **Whether `gpu="A10G"` is still accepted** as an alias for `A10`. The GPU guide omits it; an official example and the pricing FAQ still use the name.
- **Volumes v2 status.** The guide recommends v2; the CLI reference calls the flag experimental. Not resolved.
- **Whether the old `cortexdj-deap` Volume still exists** in the owner's workspace, and its version. Not checked, by instruction.
- **The size limit on a returned payload.** The docs say large payloads go to object storage and give no maximum. Irrelevant at EEGNet size.
- **DEAP's licence terms** on cloud storage were not read.
- **The Starter plan's credit and concurrency figures** are from the public pricing page; the owner's actual plan and remaining credit were not checked.
