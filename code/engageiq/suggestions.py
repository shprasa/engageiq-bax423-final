"""LLM-assisted engagement suggestions with template fallback."""
from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests

from .data_utils import _safe_int, display_summary, display_title
from .secrets import load_env


def _template_suggestion(row: pd.Series) -> str:
    src = str(row.get("source", "")).lower()
    headline = display_title(row)
    community = str(row.get("community") or row.get("domain") or "")

    if src == "github":
        if _safe_int(row.get("good_first_issue")) == 1:
            return (
                f"Open the issue \"{headline[:80]}\" in {community}, comment to confirm scope, "
                "then submit a small focused PR (docs, test, or UI fix)."
            )
        return (
            f"Visit {community}, read the README and recent issues, pick one small improvement "
            "(documentation, test coverage, or bug fix), and open a PR with a clear description."
        )
    if src == "hackernews":
        pts = int(float(row.get("upvotes") or 0))
        return (
            f"Read \"{headline[:80]}\" ({pts:,} pts), open the HN discussion, and post a "
            "5–8 sentence comment with one practical takeaway and a follow-up question."
        )
    return f"Review \"{headline[:80]}\" and decide whether to engage based on your profile fit."


def openai_configured() -> bool:
    load_env()
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def generate_suggestion(row: pd.Series, interest: str = "", cache: dict[str, str] | None = None) -> str:
    """Return LLM suggestion when OPENAI_API_KEY is set; otherwise rule-based template."""
    key = str(row.get("id", row.get("url", "")))
    if cache is not None and key in cache:
        return cache[key]

    if not openai_configured():
        out = _template_suggestion(row)
        if cache is not None:
            cache[key] = out
        return out

    headline = display_title(row)
    summary = display_summary(row)[:500]
    src = str(row.get("source", ""))
    domain = str(row.get("domain", ""))
    prompt = (
        f"You are an engagement coach. Given this opportunity and user interest profile, "
        f"write ONE specific 2–3 sentence action plan (what to read, write, or contribute).\n\n"
        f"User interests: {interest[:400]}\n"
        f"Source: {src}\nDomain: {domain}\nTitle: {headline}\nSummary: {summary}\n"
    )

    try:
        load_env()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Be concise and actionable. No bullet lists."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 180,
                "temperature": 0.4,
            },
            timeout=20,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if text:
            if cache is not None:
                cache[key] = text
            return text
    except Exception:
        pass

    out = _template_suggestion(row)
    if cache is not None:
        cache[key] = out
    return out
