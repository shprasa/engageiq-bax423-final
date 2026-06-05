from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    code_dir = Path(__file__).resolve().parent.parent
    load_dotenv(code_dir / ".env")


def require_github_token() -> str:
    load_env()
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN is missing. Copy .env.example to .env and add your token."
        )
    return token


def openai_configured() -> bool:
    load_env()
    return bool(os.getenv("OPENAI_API_KEY", "").strip())
