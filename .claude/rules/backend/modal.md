# Modal Rules and Guidelines for LLMs

This file provides rules and guidelines for LLMs when implementing Modal code in cortexdj.

The bulk of this file is Modal's own LLM guidance from <https://modal.com/llms-full.txt>. A pinned local snapshot of the full Modal docs lives at `docs/modal-llms-full.txt` — Grep / Read it for current syntax of any Modal symbol before writing code. Refresh with:

```bash
curl -sSL https://modal.com/llms-full.txt -o docs/modal-llms-full.txt
```

---

## General

- Modal is a serverless cloud platform for running Python code with minimal configuration
- Designed for AI/ML workloads but supports general-purpose cloud compute
- Serverless billing model - you only pay for resources used

## Modal documentation

- Extensive documentation is available at: modal.com/docs (and in markdown format at modal.com/llms-full.txt)
- A large collection of examples is available at: modal.com/docs/examples (and github.com/modal-labs/modal-examples)
- Reference documentation is available at: modal.com/docs/reference

Always refer to documentation and examples for up-to-date functionality and exact syntax.

## Core Modal concepts

### App

- A group of functions, classes and sandboxes that are deployed together.

### Function

- The basic unit of serverless execution on Modal.
- Each Function executes in its own container, and you can configure different Images for different Functions within the same App:

  ```python
  image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "transformers")
    .apt_install("ffmpeg")
    .run_commands("mkdir -p /models")
  )

  @app.function(image=image)
  def square(x: int) -> int:
    return x * x
  ```

- You can configure individual hardware requirements (CPU, memory, GPUs, etc.) for each Function.

  ```python
  @app.function(
    gpu="H100",
    memory=4096,
    cpu=2,
  )
  def inference():
    ...
  ```

  Some examples specifically for GPUs:

  ```python
  @app.function(gpu="A10G")  # Single GPU, e.g. T4, A10G, A100, H100, or "any"
  @app.function(gpu="A100:2")  # Multiple GPUs, e.g. 2x A100 GPUs
  @app.function(gpu=["H100", "A100", "any"]) # GPU with fallbacks
  ```

- Functions can be invoked in a number of ways. Some of the most common are:
  - `foo.remote()` - Run the Function in a separate container in the cloud. This is by far the most common.
  - `foo.local()` - Run the Function in the same context as the caller. Note: This does not necessarily mean locally on your machine.
  - `foo.map()` - Parallel map over a set of inputs.
  - `foo.spawn()` - Calls the function with the given arguments, without waiting for the results. Terminating the App will also terminate spawned functions.
- Web endpoint: You can turn any Function into an HTTP web endpoint served by adding a decorator:

  ```python
  @app.function()
  @modal.fastapi_endpoint()
  def fastapi_endpoint():
    return {"status": "ok"}

  @app.function()
  @modal.asgi_app()
  def asgi_app():
    app = FastAPI()
    ...
    return app
  ```

- You can run Functions on a schedule using e.g. `@app.function(schedule=modal.Period(minutes=5))` or `@app.function(schedule=modal.Cron("0 9 * * *"))`.

### Classes (a.k.a. `Cls`)

- For stateful operations with startup/shutdown lifecycle hooks. Example:

  ```python
  @app.cls(gpu="A100")
  class ModelServer:
      @modal.enter()
      def load_model(self):
          # Runs once when container starts
          self.model = load_model()

      @modal.method()
      def predict(self, text: str) -> str:
          return self.model.generate(text)

      @modal.exit()
      def cleanup(self):
          # Runs when container stops
          cleanup()
  ```

### Other important concepts

- Image: Represents a container image that Functions can run in.
- Sandbox: Allows defining containers at runtime and securely running arbitrary code inside them.
- Volume: Provide a high-performance distributed file system for your Modal applications.
- Secret: Enables securely providing credentials and other sensitive information to your Modal Functions.
- Dict: Distributed key/value store, managed by Modal.
- Queue: Distributed, FIFO queue, managed by Modal.

## Differences from standard Python development

- Modal always executes code in the cloud, even while you are developing. You can use Environments for separating development and production deployments.
- Dependencies: It's common and encouraged to have different dependency requirements for different Functions within the same App. Consider defining dependencies in Image definitions (see Image docs) that are attached to Functions, rather than in global `requirements.txt`/`pyproject.toml` files, and putting `import` statements inside the Function `def`. Any code in the global scope needs to be executable in all environments where that App source will be used (locally, and any of the Images the App uses).

## Modal coding style

- Modal Apps, Volumes, and Secrets should be named using kebab-case.
- Always use `import modal`, and qualified names like `modal.App()`, `modal.Image.debian_slim()`.
- Modal evolves quickly, and prints helpful deprecation warnings when you `modal run` an App that uses deprecated features. When writing new code, never use deprecated features.

## Common commands

Running `modal --help` gives you a list of all available commands. All commands also support `--help` for more details.

### Running your Modal app during development

- `modal run path/to/your/app.py` - Run your app on Modal.
- `modal run -m module.path.to.app` - Run your app on Modal, using the Python module path.
- `modal serve modal_server.py` - Run web endpoint(s) associated with a Modal app, and hot-reload code on changes. Will print a URL to the web endpoint(s). Note: you need to use `Ctrl+C` to interrupt `modal serve`.

### Deploying your Modal app

- `modal deploy path/to/your/app.py` - Deploy your app (Functions, web endpoints, etc.) to Modal.
- `modal deploy -m module.path.to.app` - Deploy your app to Modal, using the Python module path.

Logs:

- `modal app logs <app_name>` - Stream logs for a deployed app. Note: you need to use `Ctrl+C` to interrupt the stream.

### Resource management

- There are CLI commands for interacting with resources like `modal app list`, `modal volume list`, and similarly for `secret`, `dict`, `queue`, etc.
- These also support other command than `list` - use e.g. `modal app --help` for more.

## Testing and debugging

- When using `app.deploy()`, you can wrap it in a `with modal.enable_output():` block to get more output.

---

## Cortexdj-specific facts

These are project facts that aren't derivable from Modal's general docs. Treat them as ground truth:

- **The Modal entrypoint is `backend/scripts/modal_train.py`.** It runs CBraMod or EEGNet training on a GPU. All Modal CLI commands run from the **repo root**, never from `backend/`.
- **DEAP data lives in the `cortexdj-deap` Modal Volume**, not local disk for remote runs. The local copy at `backend/data/deap/` is only used for the one-time `modal volume put` seed. Reference: see the script's module docstring for the seed loop.
- **`backend/data/deap/.cache/*.npz` is regenerable preprocessing output** produced by `backend/src/cortexdj/ml/dataset.py`. **Never upload it to the volume** — it's rebuilt inside the container on first use, then committed back via `volume.commit()` at the end of the training method.
- **`backend/pyproject.toml` declares `readme = "../README.md"`**, so the Modal image needs `/app/README.md` present at build time (hatchling reads it during `uv sync`). The script handles this with `Image.add_local_file(str(REPO_ROOT / "README.md"), "/app/README.md", copy=True)`. Don't remove that line.
- **Image base must mirror `backend/Dockerfile`**: `from_registry("ghcr.io/astral-sh/uv:python3.13-bookworm-slim", add_python=None)` + `.env({"UV_COMPILE_BYTECODE": "1", "UV_LINK_MODE": "copy"})` + `uv sync --frozen --no-dev`. Diverging will produce inconsistencies between local Docker and Modal builds.
- **Training is self-contained** — no Modal `Secret`s required. If you find yourself reaching for one, double-check whether `train.py` actually needs the env var.
- **`modal volume put` and large datasets**: bulk uploads of multi-GB directories race the ~1hr S3 presigned-URL TTL on residential uplinks. For DEAP-scale data, upload one file per invocation in a bash loop (idempotent on re-runs). The script's setup docstring has the canonical loop.
- **Modal client is a regular backend dep** (`modal>=1.4.1` in `backend/pyproject.toml`). Run `modal setup` once to authenticate; do not `pip install modal` separately.

## Reference

- Pinned local docs snapshot: `docs/modal-llms-full.txt` (Grep first, WebFetch as fallback)
- Live full docs: <https://modal.com/llms-full.txt>
- Live docs index: <https://modal.com/llms.txt>
- Developing with LLMs guide: <https://modal.com/docs/guide/developing-with-llms>
