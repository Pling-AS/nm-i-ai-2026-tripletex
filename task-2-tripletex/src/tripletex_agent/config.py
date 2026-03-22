from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    openrouter_api_key: str = ""
    openrouter_model: str = "claude-sonnet-4-6"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Azure provider config (fallback for Claude models)
    azure_api_key: str = ""
    azure_openai_base_url: str = (
        "https://magnus-3458-resource.services.ai.azure.com/openai/v1"
    )
    azure_anthropic_base_url: str = (
        "https://magnus-3458-resource.services.ai.azure.com/anthropic/v1"
    )

    # Google Cloud Vertex AI config (primary backend for Claude models)
    vertex_ai_enabled: bool = True
    vertex_ai_project_id: str = ""
    vertex_ai_region: str = "europe-west1"
    vertex_ai_gcloud_bin: str = "gcloud"
    vertex_ai_token_ttl_seconds: int = 3300  # cache token ~55 min (expires ~60)

    # Multi-model routing: planner + tiered executor models
    planner_model: str = "gpt-5.4"
    planner_fast_model: str = "claude-sonnet-4-6"
    tier1_executor_model: str = "claude-sonnet-4-6"
    tier2_executor_model: str = "claude-opus-4-6"
    tier3_executor_model: str = "gpt-5.4"
    enforcer_model: str = "claude-sonnet-4-6"
    openrouter_site_url: str | None = None
    openrouter_app_name: str = "Tripletex Accounting Agent"
    tripletex_sandbox_login_email: str | None = None
    tripletex_sandbox_api_url: str | None = None
    tripletex_sandbox_api_session_token: str | None = None
    local_solve_url: str = "http://127.0.0.1:8000/solve"
    ainm_jwt_token: str = ""
    datalab_api_key: str = ""
    ainm_tripletex_task_id: str = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    app_api_key: str | None = None
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    agent_max_steps: int = 40
    agent_model_temperature: float = 0.0
    http_timeout_seconds: float = 60.0
    max_attachment_text_chars: int = 12000
    tripletex_api_spec_path: Path = Field(
        default=BASE_DIR / "task_docs" / "api_spec.json"
    )

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", extra="ignore"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
