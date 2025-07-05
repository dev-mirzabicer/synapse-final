"""Application settings and configuration."""

from typing import Optional
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Application-wide settings."""

    # API Keys
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY")

    # Logging
    log_level: str = Field("DEBUG", env="LOG_LEVEL")
    log_file: Optional[Path] = Field(None, env="LOG_FILE")
    json_logs: bool = Field(False, env="JSON_LOGS")

    # Graph Configuration
    max_iterations: int = Field(50, env="MAX_ITERATIONS")
    timeout_seconds: int = Field(300, env="TIMEOUT_SECONDS")

    # Model Configuration
    default_model: str = Field("gemini-2.5-flash", env="DEFAULT_MODEL")
    default_temperature: float = Field(0.7, env="DEFAULT_TEMPERATURE")

    # Retry Configuration
    max_retries: int = Field(3, env="MAX_RETRIES")
    retry_delay: float = Field(1.0, env="RETRY_DELAY")

    class Config:
        env_file = ".env"
        case_sensitive = False

    @field_validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


# Global settings instance
settings = AppSettings()
