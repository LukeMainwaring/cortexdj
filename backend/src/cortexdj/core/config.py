from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from cortexdj.core.paths import _BACKEND_ROOT, CHECKPOINTS_DIR, DEAP_DATA_DIR


class ApiSettings(BaseSettings):
    API_PREFIX: str = "/api"
    ALLOWED_ORIGINS: dict[str, list[str]] = {
        "development": ["http://localhost:3003"],
        "production": [],
    }


class AgentSettings(BaseSettings):
    AGENT_MODEL: str = "gpt-5.4-mini"
    AGENT_REASONING_EFFORT: Literal["low", "medium", "high"] | None = None


class PostgresSettings(BaseSettings):
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int


class SpotifySettings(BaseSettings):
    SPOTIFY_CLIENT_ID: str = ""
    SPOTIFY_CLIENT_SECRET: str = ""
    SPOTIFY_REDIRECT_URI: str = "http://127.0.0.1:8003/api/spotify/callback"
    FRONTEND_URL: str = "http://localhost:3003"


class Settings(
    ApiSettings,
    AgentSettings,
    PostgresSettings,
    SpotifySettings,
):
    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_ROOT.parent / ".env"),
        env_ignore_empty=True,
        extra="ignore",
    )

    ENVIRONMENT: Literal["development", "production"] = "development"
    EEG_MODEL_BACKEND: Literal["eegnet", "cbramod"] = "cbramod"
    DEAP_DATA_DIR: str = str(DEAP_DATA_DIR)
    CHECKPOINTS_DIR: str = str(CHECKPOINTS_DIR)

    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def spotify_configured(self) -> bool:
        return bool(self.SPOTIFY_CLIENT_ID and self.SPOTIFY_CLIENT_SECRET)


@lru_cache()
def get_settings() -> Settings:
    return Settings.model_validate({})
