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


def reddit_configured() -> bool:
    load_env()
    return all(
        os.getenv(k, "").strip()
        for k in (
            "REDDIT_CLIENT_ID",
            "REDDIT_CLIENT_SECRET",
            "REDDIT_USER_AGENT",
            "REDDIT_USERNAME",
            "REDDIT_PASSWORD",
        )
    )


def get_reddit_config() -> dict[str, str]:
    load_env()
    keys = (
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USER_AGENT",
        "REDDIT_USERNAME",
        "REDDIT_PASSWORD",
    )
    cfg = {k.lower().replace("reddit_", ""): os.getenv(k, "").strip() for k in keys}
    if not all(cfg.values()):
        raise RuntimeError("Reddit credentials incomplete in .env")
    return {
        "client_id": cfg["client_id"],
        "client_secret": cfg["client_secret"],
        "user_agent": cfg["user_agent"],
        "username": cfg["username"],
        "password": cfg["password"],
    }
