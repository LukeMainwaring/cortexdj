"""Application configuration using Pydantic Settings."""

import pathlib
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.parent.parent


class ApiSettings(BaseSettings):
    """API and CORS configuration."""

    API_PREFIX: str = "/api"
    ALLOWED_ORIGINS: dict[str, list[str]] = {
        "development": ["http://localhost:3003"],
        "production": [],
    }


class AgentSettings(BaseSettings):
    """Pydantic AI agent configuration."""

    AGENT_MODEL: str = "gpt-4o-mini"


class PostgresSettings(BaseSettings):
    """PostgreSQL connection configuration."""

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int


class SpotifySettings(BaseSettings):
    """Spotify OAuth configuration (optional)."""

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
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_ignore_empty=True,
        extra="ignore",
    )

    ENVIRONMENT: Literal["development", "production"] = "development"
    SYNTHETIC_DATA_DIR: str = str(_PROJECT_ROOT / "data" / "synthetic")
    CHECKPOINTS_DIR: str = str(_PROJECT_ROOT / "data" / "checkpoints")

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    @property
    def spotify_configured(self) -> bool:
        """Check if Spotify credentials are set."""
        return bool(self.SPOTIFY_CLIENT_ID and self.SPOTIFY_CLIENT_SECRET)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings.model_validate({})
