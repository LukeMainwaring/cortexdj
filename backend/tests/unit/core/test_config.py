"""Settings behavior: the environment flag and per-agent model resolution."""

import pytest

from cortexdj.core import config as core_config
from cortexdj.core.config import get_settings


def test_default_environment_is_development() -> None:
    settings = get_settings()
    assert settings.ENVIRONMENT == "development"
    assert settings.is_production() is False


def test_is_production_true_only_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("POSTGRES_PASSWORD", "a-real-database-password")
    get_settings.cache_clear()
    try:
        assert get_settings().is_production() is True
    finally:
        get_settings.cache_clear()  # don't leak the production-env settings to other tests


def test_production_refuses_the_development_database_password(monkeypatch: pytest.MonkeyPatch) -> None:
    # The default is in the repo: anyone who can reach Postgres could log in with it.
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("POSTGRES_PASSWORD", "postgres")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="POSTGRES_PASSWORD"):
            get_settings()
    finally:
        get_settings.cache_clear()


def test_logfire_token_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    # No token must mean "traces stay local", never a boot failure. (tests/conftest.py
    # already keeps the repo .env, and any token in it, out of every test.)
    monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)
    get_settings.cache_clear()
    try:
        assert get_settings().LOGFIRE_TOKEN is None
    finally:
        get_settings.cache_clear()


def test_an_agent_without_its_own_model_setting_follows_agent_model() -> None:
    settings = core_config.AgentSettings(AGENT_MODEL="anthropic:claude-sonnet-5-5", THREAD_TITLE_MODEL=None)
    assert settings.resolve_model(settings.THREAD_TITLE_MODEL) == "anthropic:claude-sonnet-5-5"


def test_an_agent_with_its_own_model_setting_uses_it() -> None:
    settings = core_config.AgentSettings(
        AGENT_MODEL="anthropic:claude-sonnet-5-5", THREAD_TITLE_MODEL="anthropic:claude-haiku-5-5"
    )
    assert settings.resolve_model(settings.THREAD_TITLE_MODEL) == "anthropic:claude-haiku-5-5"


def test_the_eval_judge_model_is_pinned_apart_from_agent_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)
    settings = core_config.AgentSettings(AGENT_MODEL="openai:gpt-6.1-sol")
    assert settings.EVAL_JUDGE_MODEL == "anthropic:claude-opus-5-5"
