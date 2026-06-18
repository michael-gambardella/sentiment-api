"""Centralised runtime configuration via pydantic-settings.

Values are read from environment variables or a .env file in the project root.
All fields have sensible defaults so the app runs out-of-the-box with no setup.
"""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: str = Field(
        default="development",
        description="Runtime environment: development | staging | production",
    )
    log_level: str = Field(
        default="INFO",
        description="Python logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL",
    )
    model_name: str = Field(
        default="distilbert-base-uncased",
        description="HuggingFace model identifier used for tokenisation and base weights",
    )
    artifacts_dir: Path = Field(
        default=Path("model/artifacts/final"),
        description="Path to the fine-tuned model artifact directory (relative to project root or absolute)",
    )
    max_input_length: int = Field(
        default=256,
        description="Maximum token sequence length; inputs beyond this are truncated",
    )
    batch_size: int = Field(
        default=16,
        description="DataLoader batch size used during training and evaluation",
    )
    api_keys: str = Field(
        default="",
        description="Comma-separated valid API keys for /predict. Empty = auth disabled.",
    )
    rate_limit: str = Field(
        default="60/minute",
        description="Request rate limit for /predict (e.g. '60/minute', '10/second'). Uses slowapi syntax.",
    )
    redis_url: str = Field(
        default="redis://redis:6379/0",
        description="Redis connection URL used as the Celery broker and result backend.",
    )


settings = Settings()
