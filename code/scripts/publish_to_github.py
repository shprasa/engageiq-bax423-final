from __future__ import annotations

import base64
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
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from engageiq.secrets import require_github_token

OWNER = "shprasa"
REPO = "engageiq-bax423-final"
DESCRIPTION = "EngageIQ — BAX-423 Final: Smart Engagement Opportunity Scorer (UC Davis MSBA)"
ROOT = CODE_DIR.parent

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "_tmp_engageiq_docx"}
SKIP_FILES = {".env", ".DS_Store"}
SKIP_SUFFIXES = {".duckdb", ".duckdb.wal", ".pyc"}


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def create_repo(token: str) -> None:
    url = "https://api.github.com/user/repos"
    payload = {
        "name": REPO,
        "description": DESCRIPTION,
        "private": False,
        "auto_init": False,
    }
    r = requests.post(url, headers=_headers(token), json=payload, timeout=30, verify=certifi.where())
    if r.status_code == 422 and "already exists" in r.text.lower():
        print("Repo already exists — continuing upload.")
        return
    r.raise_for_status()
    print("Created repo:", r.json().get("html_url"))


def upload_file(token: str, rel_path: str, content: bytes, message: str) -> None:
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{rel_path}"
    r = requests.get(url, headers=_headers(token), timeout=30, verify=certifi.where())
    sha = r.json().get("sha") if r.status_code == 200 else None

    body = {
        "message": message,
        "content": base64.b64encode(content).decode("ascii"),
    }
    if sha:
        body["sha"] = sha

    r = requests.put(url, headers=_headers(token), json=body, timeout=60, verify=certifi.where())
    if r.status_code >= 400:
        raise RuntimeError(f"Upload failed {rel_path}: {r.status_code} {r.text[:200]}")


def iter_files() -> list[Path]:
    files: list[Path] = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.name in SKIP_FILES or p.suffix in SKIP_SUFFIXES:
            continue
        # skip very large csv if > 25MB - github limit per file via API is strict
        if p.stat().st_size > 24 * 1024 * 1024:
            print("SKIP large file:", p)
            continue
        files.append(p)
    return files


def main() -> None:
    token = require_github_token()
    create_repo(token)

    files = iter_files()
    print(f"Uploading {len(files)} files to {OWNER}/{REPO} ...")
    for i, path in enumerate(sorted(files), start=1):
        rel = path.relative_to(ROOT).as_posix()
        try:
            content = path.read_bytes()
        except OSError as e:
            print("SKIP unreadable:", rel, e)
            continue
        upload_file(token, rel, content, f"Add {rel}")
        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)} uploaded")
    print(f"Done: https://github.com/{OWNER}/{REPO}")


if __name__ == "__main__":
    main()
