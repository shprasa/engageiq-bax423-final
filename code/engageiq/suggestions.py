"""Engagement suggestions — free LLM APIs (Gemini/Groq) or smart templates (capability 6)."""
from __future__ import annotations

import os
import re

import certifi
import pandas as pd
import requests

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from .data_utils import _is_github_issue, _safe_int, display_summary, display_title
from .secrets import load_env

_LAST_PROVIDER: str = "template"


def _ssl_verify() -> bool | str:
    if os.getenv("ENGAGEIQ_SSL_VERIFY", "true").lower() in {"0", "false", "no"}:
        return False
    return certifi.where()


def _interest_keywords(interest: str, limit: int = 4) -> list[str]:
    words = re.findall(r"[A-Za-z0-9+#./-]{3,}", interest.lower())
    stop = {"and", "the", "for", "with", "from", "your", "that", "this", "have", "news"}
    out: list[str] = []
    for w in words:
        if w in stop or w in out:
            continue
        out.append(w)
        if len(out) >= limit:
            break
    return out


def _issue_ref(url: str) -> str:
    url = url.rstrip("/")
    if "/issues/" in url:
        return f"issue #{url.split('/issues/')[-1].split('/')[0]}"
    if "/pull/" in url:
        return f"PR #{url.split('/pull/')[-1].split('/')[0]}"
    return "this thread"


def _template_suggestion(row: pd.Series, interest: str = "") -> str:
    """Interest-aware rule-based plan — no API cost, always available."""
    src = str(row.get("source", "")).lower()
    headline = display_title(row)
    community = str(row.get("community") or row.get("domain") or "this community")
    domain = str(row.get("domain") or "")
    url = str(row.get("url") or "")
    summary = display_summary(row)[:100].strip()
    kws = _interest_keywords(interest) or [domain.split("/")[0].lower()]
    focus = kws[0]

    if src == "github":
        stars = _safe_int(row.get("stars"))
        gfi = _safe_int(row.get("good_first_issue")) == 1
        if gfi or _is_github_issue(row):
            hint = summary or headline[:80]
            ref = _issue_ref(url)
            return (
                f"Start with \"{headline[:62]}\" ({community}, {ref}): read the thread"
                f"{f' — {hint}…' if hint else ''}, post one clarifying question, "
                f"then open a small docs/tests PR that closes or references {ref}."
            )
        return (
            f"Explore {community} ({domain}, {stars:,}★): clone the repo, read CONTRIBUTING.md, "
            f"and use \"{headline[:52]}\" as your entry point — pick a {focus}-related docs or "
            "test issue and open a PR with a one-paragraph maintainer note."
        )
    if src == "gharchive":
        com = _safe_int(row.get("comments"))
        kind = "issue thread" if "/issues/" in url else ("PR thread" if "/pull/" in url else "timeline event")
        return (
            f"Archive {kind}: \"{headline[:58]}\" on {community} ({com:,} comments, {domain}). "
            f"Open the live URL, scan the last 5 comments for {focus} context, and reply with "
            f"one concrete suggestion or link a minimal follow-up PR for {_issue_ref(url)}."
        )
    return (
        f"Check \"{headline[:65]}\" ({domain}): only engage if it advances your {focus} goals — "
        "leave one specific comment or contribution, not a generic star/follow."
    )


def _set_provider(name: str) -> None:
    global _LAST_PROVIDER
    _LAST_PROVIDER = name


def suggestion_provider() -> str:
    """Last provider used: gemini, groq, openai, or template."""
    return _LAST_PROVIDER


def llm_configured() -> bool:
    load_env()
    return bool(
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GROQ_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )


def openai_configured() -> bool:
    """Backward-compatible alias."""
    return llm_configured()


def _call_gemini(prompt: str) -> str | None:
    load_env()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    preferred = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()
    fallbacks = [
        preferred,
        "gemini-2.5-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash",
    ]
    seen: set[str] = set()
    for model in fallbacks:
        if not model or model in seen:
            continue
        seen.add(model)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        resp = requests.post(
            url,
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=25,
            verify=_ssl_verify(),
        )
        if resp.status_code != 200:
            continue
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            continue
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = str(parts[0].get("text", "")).strip() if parts else ""
        if text:
            return text
    return None


def _call_groq(prompt: str) -> str | None:
    load_env()
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
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
        timeout=25,
        verify=_ssl_verify(),
    )
    if resp.status_code != 200:
        return None
    text = resp.json()["choices"][0]["message"]["content"].strip()
    return text or None


def _call_openai(prompt: str) -> str | None:
    load_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
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
        timeout=25,
        verify=_ssl_verify(),
    )
    if resp.status_code != 200:
        return None
    text = resp.json()["choices"][0]["message"]["content"].strip()
    return text or None


def _build_prompt(row: pd.Series, interest: str) -> str:
    headline = display_title(row)
    summary = display_summary(row)[:500]
    src = str(row.get("source", ""))
    domain = str(row.get("domain", ""))
    community = str(row.get("community") or "")
    stars = _safe_int(row.get("stars"))
    return (
        "You are an engagement coach. Given this opportunity and user interest profile, "
        "write ONE specific 2–3 sentence action plan (what to read, write, or contribute). "
        "You MUST name the opportunity title or repo/community in the first sentence. "
        "Do not use generic filler like 'looking for insights'.\n\n"
        f"User interests: {interest[:400]}\n"
        f"Source: {src}\nDomain: {domain}\nCommunity: {community}\nStars: {stars}\n"
        f"Title: {headline}\nSummary: {summary}\n"
    )


def _suggestion_cache_key(row: pd.Series, interest: str) -> str:
    rid = str(row.get("id", row.get("url", "")))
    profile = (interest or "").strip()[:160]
    return f"{rid}|{hash(profile)}"


def generate_suggestion(row: pd.Series, interest: str = "", cache: dict[str, str] | None = None) -> str:
    key = _suggestion_cache_key(row, interest)
    if cache is not None and key in cache:
        return cache[key]

    prompt = _build_prompt(row, interest)
    providers: list[tuple[str, callable]] = [
        ("gemini", _call_gemini),
        ("groq", _call_groq),
        ("openai", _call_openai),
    ]

    for name, fn in providers:
        try:
            text = fn(prompt)
            if text:
                _set_provider(name)
                if cache is not None:
                    cache[key] = text
                return text
        except Exception:
            continue

    _set_provider("template")
    out = _template_suggestion(row, interest)
    if cache is not None:
        cache[key] = out
    return out
