from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py -> core -> cortexdj -> src -> backend -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]

# The password .env.sample ships; fine for a local Compose stack, never for production.
DEV_POSTGRES_PASSWORD = "postgres"


class ApiSettings(BaseSettings):
    API_PREFIX: str = "/api"
    ALLOWED_ORIGINS: dict[str, list[str]] = {
        "development": ["http://localhost:3003"],
        "production": [],
    }


class AgentSettings(BaseSettings):
    # Provider-agnostic model identifier. The prefix selects the provider:
    # "anthropic:..." for Anthropic, "openai:..." for OpenAI. pydantic-ai
    # infers the provider client from this prefix at agent construction time.
    AGENT_MODEL: str = "anthropic:claude-sonnet-5-5"
    # Applies to every agent built with build_agent. A model pydantic-ai has not
    # profiled yet may get the wrong thinking request shape and a 400 from the
    # provider; the fix is to upgrade pydantic-ai, not to patch profiles.
    AGENT_REASONING_EFFORT: Literal["minimal", "low", "medium", "high", "xhigh", "max"] | None = None
    # Per-agent models: None follows AGENT_MODEL. Thread titles default to Haiku because
    # a 3-6 word summary needs no bigger model. The default is on the same provider as
    # AGENT_MODEL's; a deployment on another provider sets this too (a blank env value
    # means unset, not "follow AGENT_MODEL").
    THREAD_TITLE_MODEL: str | None = "anthropic:claude-haiku-5-5"

    def resolve_model(self, agent_model: str | None) -> str:
        """The model an agent runs on: its own setting (e.g. ``THREAD_TITLE_MODEL``) when set, else ``AGENT_MODEL``."""
        return agent_model or self.AGENT_MODEL

    # The model the eval judges (tests/evals/judges.py) grade replies on. Pinned: it does not
    # follow AGENT_MODEL, so changing an agent's model never changes who grades it, and it is a
    # tier above the agents' default rather than one of them (ADR 0004). The default needs
    # ANTHROPIC_API_KEY even when the agents run elsewhere; a deployment on another provider
    # sets this too. Run the golden set (tests/evals/golden_set.py) after changing it.
    EVAL_JUDGE_MODEL: str = "anthropic:claude-opus-5-5"


class ObservabilitySettings(BaseSettings):
    # Read through Settings, not only from the process environment, so a token in the
    # repo .env reaches a bare `uv run` as well as Compose. Unset, nothing is exported.
    LOGFIRE_TOKEN: SecretStr | None = None


class PostgresSettings(BaseSettings):
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int


class Settings(
    ApiSettings,
    AgentSettings,
    ObservabilitySettings,
    PostgresSettings,
):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_ignore_empty=True,
        extra="ignore",
    )

    ENVIRONMENT: Literal["development", "production"] = "development"

    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @model_validator(mode="after")
    def _require_a_real_database_password_in_production(self) -> Self:
        # The default is in the repo, so with it anyone who can reach Postgres logs in.
        if self.is_production() and self.POSTGRES_PASSWORD == DEV_POSTGRES_PASSWORD:
            raise ValueError(
                "POSTGRES_PASSWORD is the development value from .env.sample; set a real one in production"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings.model_validate({})
