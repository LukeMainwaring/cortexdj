"""Suite-wide isolation: tests see a fixed environment, never the developer's ``.env`` or a real provider.

Everything here runs at import, before any test module imports the app, because the
app reads its settings and builds its agents at import time.
"""

import os

import pytest
from pydantic_ai import models

from cortexdj.core.config import Settings

# Any run that reaches a real provider model raises instead of spending money. TestModel
# and FunctionModel are unaffected; the eval tier opts back in (tests/evals/conftest.py).
models.ALLOW_MODEL_REQUESTS = False

# The repo-root .env carries the developer's real keys and model choice. Tests read none
# of it: settings come from the process environment and the defaults below.
Settings.model_config["env_file"] = None

# pydantic-ai constructs its provider client when an agent is built, which needs a key
# to be present. These dummies satisfy that; ALLOW_MODEL_REQUESTS stops them being used.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-deterministic-no-real-calls")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-deterministic-no-real-calls")

# Settings require Postgres connection vars at import time. The unit tier never connects;
# the integration tier connects to this isolated test database (see tests/integration/conftest.py).
os.environ.setdefault("POSTGRES_USER", "postgres")
os.environ.setdefault("POSTGRES_PASSWORD", "postgres")
os.environ.setdefault("POSTGRES_DB", "cortexdj_test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5433")


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend only."""
    return "asyncio"
