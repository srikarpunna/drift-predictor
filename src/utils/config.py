from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class GeminiModels:
    OLD = "gemini-2.5-flash"
    NEW = "gemini-3.1-flash-lite"


class ClaudeModels:
    OLD = "claude-sonnet-4-5-20250929"
    NEW = "claude-sonnet-4-6"


@dataclass
class RetryConfig:
    max_attempts: int = 3
    wait_min_seconds: float = 1.0
    wait_max_seconds: float = 10.0
    wait_multiplier: float = 2.0


@dataclass
class Settings:
    gemini_api_key: str = field(default_factory=lambda: _require_env("GEMINI_API_KEY"))
    anthropic_api_key: str = field(default_factory=lambda: _require_env("ANTHROPIC_API_KEY"))
    retry: RetryConfig = field(default_factory=RetryConfig)
    request_timeout_seconds: float = 60.0


def _require_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example to .env and fill in your API keys."
        )
    return value


settings = Settings()
