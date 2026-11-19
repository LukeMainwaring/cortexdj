import os

import pytest

# ``OpenAIResponsesModel`` constructs its provider client eagerly at module
# import time, which requires ``OPENAI_API_KEY``. Any test that imports
# ``brain_agent`` — directly or transitively — would otherwise fail at
# collection. Tests that actually invoke the model use ``agent.override``
# with ``TestModel``; this dummy value never reaches a real API call.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-deterministic-no-real-calls")

# Deliberately no POSTGRES_* defaults here: settings load from the repo-root
# ``.env`` (port 5433), and real env vars take precedence over dotenv — a
# setdefault would silently override ``.env`` and break local integration runs.


@pytest.fixture
def anyio_backend() -> str:
    """Run ``@pytest.mark.anyio`` async tests on the asyncio backend."""
    return "asyncio"
