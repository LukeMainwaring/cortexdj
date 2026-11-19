"""Observability setup is once-per-process and works with no LOGFIRE_TOKEN.

The instrumentors warn (and double-wrap) when attached twice, and ``app.py`` is
imported more than once under ``python -m`` and under uvicorn's reloader — so
idempotency is the behavior worth pinning, not the individual calls.
"""

from typing import Any

import logfire
import pytest

from cortexdj.core import observability


@pytest.fixture(autouse=True)
def _unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test from a fresh process, and never touch the real Logfire SDK."""
    monkeypatch.setattr(observability, "_configured", False)


def _record_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def record(name: str) -> Any:
        def _call(*args: Any, **kwargs: Any) -> None:
            calls.append((name, kwargs))

        return _call

    for name in ("configure", "instrument_sqlalchemy", "instrument_pydantic_ai"):
        monkeypatch.setattr(logfire, name, record(name))
    return calls


def test_configures_before_instrumenting(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_calls(monkeypatch)

    observability.configure_observability()

    # Ordering is load-bearing: instrumentors registered before configure() send nowhere.
    assert [name for name, _ in calls] == ["configure", "instrument_sqlalchemy", "instrument_pydantic_ai"]


def test_runs_without_a_logfire_token(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_calls(monkeypatch)
    monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)

    observability.configure_observability()

    assert calls[0][1]["send_to_logfire"] == "if-token-present"
    assert calls[0][1]["service_name"] == "cortexdj"


def test_second_call_does_not_reinstrument(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _record_calls(monkeypatch)

    observability.configure_observability()
    observability.configure_observability()

    assert len(calls) == 3, "a repeat import must not attach the instrumentors twice"


def test_httpx_is_not_instrumented(monkeypatch: pytest.MonkeyPatch) -> None:
    # instrument_httpx captures exact provider payloads — prompts, tool data, API keys.
    # It is a deliberate opt-in for a debugging session, never the default.
    _record_calls(monkeypatch)
    called = False

    def _fail(*args: Any, **kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(logfire, "instrument_httpx", _fail)

    observability.configure_observability()

    assert not called
