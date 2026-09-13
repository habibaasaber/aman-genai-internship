"""
config.py — Application settings loaded from environment variables / .env file.

All configuration is centralised here via pydantic-settings so that:
  1. Every value has a type, default, and documentation.
  2. The .env file is automatically read when running locally.
  3. Docker / CI/CD environments can override via real env vars.
"""

from __future__ import annotations

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ------------------------------------------------------------------ #
    # LLM — Primary (OpenAI)
    # ------------------------------------------------------------------ #
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_base_url: str = "https://api.openai.com/v1"

    # ------------------------------------------------------------------ #
    # LLM — Fallback (Google Gemini)
    # ------------------------------------------------------------------ #
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # ------------------------------------------------------------------ #
    # Retry / Backoff
    # ------------------------------------------------------------------ #
    max_retries: int = 3
    retry_base_delay: float = 1.0   # seconds; delay = base * 2^attempt
    retry_max_delay: float = 30.0   # hard cap per retry sleep

    # ------------------------------------------------------------------ #
    # GitHub MCP Server
    # ------------------------------------------------------------------ #
    github_personal_access_token: str = ""

    # ------------------------------------------------------------------ #
    # Langfuse Observability
    # ------------------------------------------------------------------ #
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = True

    # ------------------------------------------------------------------ #
    # FastAPI
    # ------------------------------------------------------------------ #
    backend_host: str = "0.0.0.0"
    backend_port: int = 8001
    # Comma-separated CORS-allowed origins (Chainlit URL).
    allowed_origins: str = "http://localhost:8000,http://chainlit:8000"

    # ------------------------------------------------------------------ #
    # Chainlit Authentication
    # ------------------------------------------------------------------ #
    chainlit_auth_secret: str = "please-change-this-to-a-strong-random-secret"
    # Format: "username1:password1,username2:password2"
    chainlit_users: str = "admin:admin123,demo:demo456"

    # URL that the Chainlit app uses to reach the FastAPI backend.
    backend_url: str = "http://localhost:8001"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def chainlit_users_dict(self) -> dict[str, str]:
        """Parse "user1:pass1,user2:pass2" into a dict."""
        users: dict[str, str] = {}
        for pair in self.chainlit_users.split(","):
            pair = pair.strip()
            if ":" in pair:
                username, _, password = pair.partition(":")
                users[username.strip()] = password.strip()
        return users

    @property
    def gemini_model_litellm(self) -> str:
        """LiteLLM expects 'gemini/<model-name>' for Gemini models."""
        if self.gemini_model.startswith("gemini/"):
            return self.gemini_model
        return f"gemini/{self.gemini_model}"


settings = Settings()
