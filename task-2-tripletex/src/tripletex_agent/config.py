from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str | None = None
    openrouter_app_name: str = "Tripletex Accounting Agent"
    tripletex_sandbox_login_email: str | None = None
    tripletex_sandbox_api_url: str | None = None
    tripletex_sandbox_api_session_token: str | None = None
    local_solve_url: str = "http://127.0.0.1:8000/solve"
    app_api_key: str | None = None
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    agent_max_steps: int = 28
    agent_model_temperature: float = 0.0
    http_timeout_seconds: float = 45.0
    max_attachment_text_chars: int = 12000
    tripletex_api_spec_path: Path = Field(
        default=BASE_DIR / "task_docs" / "api_spec.json"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
