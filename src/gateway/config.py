from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """CryptoGuard gateway configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="CG_")

    nebius_api_key: str = ""
    nim_base_url: str = "https://api.studio.nebius.com/v1/"
    model_ultra: str = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
    model_nano: str = "nvidia/llama-3.1-nemotron-nano-8b-v1"
    neva_endpoint: str = ""
    redis_url: str = ""
    rl_feedback_enabled: bool = False
    rate_limit_rpm: int = 60
    log_level: str = "INFO"
    pool_size_bytes: int = 4096
    allowed_origins: str = ""

    @property
    def is_cloud_mode(self) -> bool:
        """True when a Nebius API key is configured."""
        return bool(self.nebius_api_key)

    @property
    def is_pipeline_mode(self) -> bool:
        """True when a Redis URL is configured."""
        return bool(self.redis_url)
