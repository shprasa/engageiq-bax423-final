"""Run benchmarks and write results (used by CLI scripts and profile save)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import Paths, get_paths
from .persona_eval import build_benchmark_payload
from .profile_store import custom_profiles_path


def benchmark_results_path(paths: Paths | None = None) -> Path:
    paths = paths or get_paths()
    if paths.is_cloud:
        return Path("/tmp/engageiq_benchmark_results.json")
    return paths.data_dir / "benchmark_results.json"


def run_and_write_benchmarks(
    df: pd.DataFrame,
    paths: Paths | None = None,
    custom_path: Path | None = None,
) -> dict:
    paths = paths or get_paths()
    custom_path = custom_path or custom_profiles_path(paths)
    payload = build_benchmark_payload(df, custom_profiles_path=custom_path)
    out = benchmark_results_path(paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    out.write_text(text, encoding="utf-8")
    # Keep project data/ and code/data/ in sync (Streamlit Cloud reads code/data/).
    sync_paths = {
        paths.project_root / "data" / "benchmark_results.json",
        paths.code_dir / "data" / "benchmark_results.json",
    }
    for alt in sync_paths:
        if alt.resolve() != out.resolve():
            alt.parent.mkdir(parents=True, exist_ok=True)
            alt.write_text(text, encoding="utf-8")
    return payload


def persona_benchmark_row(payload: dict, persona_name: str) -> dict | None:
    for row in payload.get("personas", []):
        if row.get("persona") == persona_name:
            return row
    return None


def benchmark_summary(payload: dict) -> str:
    personas = payload.get("personas", [])
    passed = sum(1 for p in personas if p.get("passed"))
    custom = len(payload.get("custom_profiles", []))
    return f"Benchmarks updated: {passed}/{len(personas)} personas passed ({custom} custom)."
