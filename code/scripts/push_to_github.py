"""Push EngageIQ_Final to GitHub via REST API (no git CLI required)."""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

import certifi
import requests

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from engageiq.secrets import load_env

OWNER = "shprasa"
REPO = "engageiq-bax423-final"
BRANCH = "main"
COMMIT_MSG = "Refresh brief, prompts, rubric docs, and submission ZIP for Canvas"

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "_tmp_engageiq_docx", ".cursor"}
SKIP_NAMES = {".env", "engageiq.duckdb", "engageiq.duckdb.wal", "Thumbs.db", ".DS_Store"}
TEXT_EXT = {
    ".py", ".md", ".txt", ".toml", ".json", ".gitignore", ".example", ".template",
}
BINARY_EXT = {".pdf", ".csv", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".zip"}


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _collect_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.name in SKIP_NAMES:
            continue
        if path.suffix in {".duckdb", ".wal", ".pyc"}:
            continue
        files.append(path)
    return sorted(files)


def _file_content(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext in BINARY_EXT:
        raw = path.read_bytes()
        return base64.b64encode(raw).decode("ascii"), "base64"
    if ext in TEXT_EXT or ext == "":
        try:
            return path.read_text(encoding="utf-8"), "utf-8"
        except UnicodeDecodeError:
            pass
    raw = path.read_bytes()
    return base64.b64encode(raw).decode("ascii"), "base64"


def _api(token: str, method: str, url: str, **kwargs) -> requests.Response:
    kwargs.setdefault("verify", certifi.where())
    resp = requests.request(method, url, headers=_headers(token), timeout=120, **kwargs)
    return resp


def main() -> None:
    load_env()
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        raise SystemExit("GITHUB_TOKEN missing in code/.env")

    base = f"https://api.github.com/repos/{OWNER}/{REPO}"
    ref_resp = _api(token, "GET", f"{base}/git/ref/heads/{BRANCH}")
    if ref_resp.status_code == 404:
        raise SystemExit(f"Branch {BRANCH} not found on {OWNER}/{REPO}")
    ref_resp.raise_for_status()
    base_commit = ref_resp.json()["object"]["sha"]
    print(f"Base commit: {base_commit[:8]}")

    files = _collect_files(PROJECT_ROOT)
    print(f"Uploading {len(files)} files...")

    tree_entries: list[dict] = []
    for path in files:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        content, enc = _file_content(path)
        blob_resp = _api(
            token,
            "POST",
            f"{base}/git/blobs",
            json={"content": content, "encoding": enc},
        )
        blob_resp.raise_for_status()
        tree_entries.append(
            {
                "path": rel,
                "mode": "100644",
                "type": "blob",
                "sha": blob_resp.json()["sha"],
            }
        )
        print(f"  blob {rel}")

    tree_resp = _api(token, "POST", f"{base}/git/trees", json={"tree": tree_entries})
    tree_resp.raise_for_status()
    tree_sha = tree_resp.json()["sha"]

    commit_resp = _api(
        token,
        "POST",
        f"{base}/git/commits",
        json={
            "message": COMMIT_MSG,
            "tree": tree_sha,
            "parents": [base_commit],
        },
    )
    commit_resp.raise_for_status()
    new_commit = commit_resp.json()["sha"]
    print(f"New commit: {new_commit[:8]}")

    update_resp = _api(
        token,
        "PATCH",
        f"{base}/git/refs/heads/{BRANCH}",
        json={"sha": new_commit, "force": False},
    )
    update_resp.raise_for_status()
    print(f"Pushed to https://github.com/{OWNER}/{REPO}")
    print("Streamlit Cloud should redeploy in ~2–3 minutes.")


if __name__ == "__main__":
    main()
