"""
EngageIQ automated user-testing & rubric validation loop.

Simulates key user journeys (discover → rank → decide → engage/bookmark/skip → RL update)
and checks assignment requirements. Re-run until all checks pass:

    py scripts/user_test_loop.py
    py scripts/user_test_loop.py --loop --max-rounds 5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import pandas as pd

from engageiq.config import get_paths
from engageiq.data import OpportunityStore
from engageiq.data_utils import (
    decision_facts,
    display_subtitle,
    display_summary,
    display_title,
    estimated_engagement_time,
    is_live_url,
    live_mask,
    opportunity_type_label,
    ranking_corpus,
)
from engageiq.persona_eval import PERSONAS, evaluate_personas, learning_benchmark
from engageiq.ranking import RankConfig, augment_candidates, rerank
from engageiq.reinforcement_learning import EngagementRLAgent
from engageiq.embedding import build_index
import numpy as np


@dataclass
class Check:
    name: str
    category: str
    fn: Callable[[], None]


@dataclass
class Report:
    passed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.failed) == 0

    def to_dict(self) -> dict:
        return {
            "passed": len(self.passed),
            "failed": len(self.failed),
            "warnings": self.warnings,
            "failures": [{"check": n, "error": e} for n, e in self.failed],
            "passed_checks": self.passed,
        }


def _fail(report: Report, name: str, msg: str) -> None:
    report.failed.append((name, msg))


def _pass(report: Report, name: str) -> None:
    report.passed.append(name)


def load_dataframe() -> tuple[pd.DataFrame, OpportunityStore]:
    paths = get_paths()
    store = OpportunityStore(paths.duckdb_path, snapshot_csv=paths.snapshot_csv)
    store.ensure_loaded_from_snapshot(paths.snapshot_csv, initial_ingest=100000)
    return store.load_df(), store


def check_submission_files(report: Report) -> None:
    name = "submission_files"
    required = [
        PROJECT_ROOT / "brief.pdf",
        PROJECT_ROOT / "prompts.md",
        CODE_DIR / "requirements.txt",
        CODE_DIR / "README.md",
        PROJECT_ROOT / "data" / "benchmark_results.json",
        CODE_DIR / "data" / "live_opportunities.csv",
        CODE_DIR / "app.py",
    ]
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in required if not p.exists()]
    if missing:
        _fail(report, name, f"Missing files: {missing}")
    else:
        _pass(report, name)


def check_dataset_size(report: Report, df: pd.DataFrame) -> None:
    name = "dataset_size"
    from engageiq.domains import DOMAINS

    if len(df) < 10_000:
        _fail(report, name, f"Need >=10,000 rows, got {len(df)}")
        return
    present = set(df["domain"].dropna().astype(str).unique())
    missing = sorted(set(DOMAINS) - present)
    if missing:
        _fail(report, name, f"Missing domains: {missing}")
        return
    if df["domain"].nunique() < 15:
        _fail(report, name, f"Need >=15 domains, got {df['domain'].nunique()}")
        return
    live = int(live_mask(df).sum())
    if live < 500:
        _fail(report, name, f"Need >=500 live rows, got {live}")
        return
    by_src = df.loc[live_mask(df), "source"].value_counts()
    if len(by_src) < 2:
        _fail(report, name, f"Need >=2 live API sources, got {by_src.to_dict()}")
        return
    for src in ("github", "gharchive"):
        if int(by_src.get(src, 0)) < 100:
            _fail(report, name, f"Need >=100 live {src} rows, got {by_src.get(src, 0)}")
            return
    _pass(report, name)


def check_live_card_titles(report: Report, df: pd.DataFrame) -> None:
    name = "live_card_titles"
    live = df[live_mask(df)]
    bad: list[str] = []

    for src in ("github", "gharchive"):
        sample = live[live["source"] == src].head(50)
        for _, row in sample.iterrows():
            title = display_title(row)
            if not title or len(title) < 5:
                bad.append(f"{src} id={row['id']}: empty/short title")
            if "example.local" in str(row.get("url", "")):
                bad.append(f"{src} id={row['id']}: example.local in live row")
            if title.lower().startswith("looking for insights"):
                bad.append(f"{src} id={row['id']}: synthetic template title")

    gfi = live[(live["source"] == "github") & (live["good_first_issue"].fillna(0).astype(int) == 1)].head(20)
    for _, row in gfi.iterrows():
        t = display_title(row)
        if "/" in t and " " not in t:
            bad.append(f"GFI id={row['id']}: title looks like repo path not issue title")

    if bad:
        _fail(report, name, "; ".join(bad[:8]) + (f" (+{len(bad)-8} more)" if len(bad) > 8 else ""))
    else:
        _pass(report, name)


def check_card_decision_info(report: Report, df: pd.DataFrame) -> None:
    name = "card_decision_info"
    live = df[live_mask(df)].head(100)
    issues: list[str] = []

    for _, row in live.iterrows():
        summary = display_summary(row)
        facts = decision_facts(row)
        est = estimated_engagement_time(row)
        subtitle = display_subtitle(row)
        type_lbl = opportunity_type_label(row)

        if len(summary) < 30:
            issues.append(f"id={row['id']}: summary too short ({len(summary)} chars)")
        if len(facts) < 4:
            issues.append(f"id={row['id']}: only {len(facts)} decision facts")
        if not est:
            issues.append(f"id={row['id']}: missing effort estimate")
        if not subtitle:
            issues.append(f"id={row['id']}: missing subtitle")
        if not type_lbl:
            issues.append(f"id={row['id']}: missing type label")

    if issues:
        _fail(report, name, "; ".join(issues[:6]) + (f" (+{len(issues)-6} more)" if len(issues) > 6 else ""))
    else:
        _pass(report, name)


def check_live_only_ranking(report: Report, df: pd.DataFrame) -> None:
    name = "live_only_ranking"
    corpus = ranking_corpus(df, live_only=True, english_only=True)
    if len(corpus) < 100:
        _fail(report, name, f"English live corpus too small: {len(corpus)}")
        return
    if not live_mask(corpus).all():
        fake = int((~live_mask(corpus)).sum())
        _fail(report, name, f"Live-only corpus contains {fake} non-live URLs")
        return
    _pass(report, name)


def check_plain_text_summaries(report: Report, df: pd.DataFrame) -> None:
    name = "plain_text_summaries"
    from engageiq.data_utils import strip_markdown

    live = df[live_mask(df)].head(200)
    bad = 0
    for _, row in live.iterrows():
        summary = display_summary(row)
        if "###" in summary or summary.startswith("#"):
            bad += 1
        if strip_markdown(str(row.get("text") or "")) != strip_markdown(str(row.get("text") or "")):
            pass
    if bad > 0:
        _fail(report, name, f"{bad} summaries still contain markdown heading markers")
    else:
        _pass(report, name)


def check_persona_benchmarks(report: Report, df: pd.DataFrame) -> None:
    name = "persona_benchmarks"
    learning = learning_benchmark(df, PERSONAS["Raj (Startup Founder / Marketing-Focused)"], rounds=60)
    learning_ok = learning["improvement"] > 0 or learning["reward_improvement_last10"] > 0
    results = evaluate_personas(df, learning_ok=learning_ok)

    persona_keys = {
        "Sofia (ML Student / Portfolio Builder)": "pass_sofia",
        "David (DevOps / Niche Community)": "pass_david",
        "Lina (Data Journalist / Trend Spotter)": "pass_lina",
        "Raj (Startup Founder / Marketing-Focused)": "pass_raj",
    }
    errors: list[str] = []
    for r in results:
        key = persona_keys.get(r.persona)
        if key and not getattr(r, key):
            errors.append(f"{r.persona} failed {key}")
        partial = [k for k, v in r.capability_pass.items() if v != "PASS"]
        if partial:
            errors.append(f"{r.persona} partial capabilities: {partial}")

    if learning["rounds"] < 50:
        errors.append(f"RL rounds {learning['rounds']} < 50")

    if errors:
        _fail(report, name, "; ".join(errors))
    else:
        _pass(report, name)


def simulate_user_session(report: Report, df: pd.DataFrame) -> None:
    """Simulate discover → engage/bookmark/skip → RL learning."""
    name = "user_session_simulation"
    try:
        corpus = ranking_corpus(df, live_only=True)
        interest = PERSONAS["Sofia (ML Student / Portfolio Builder)"]
        index = build_index(corpus)
        idxs, dists = index.query(interest, top_k=RankConfig().candidate_k)
        candidates = corpus.iloc[idxs].copy().reset_index(drop=True)
        candidates = augment_candidates(candidates, corpus, interest)
        rel = np.clip(1.0 - np.asarray(dists, dtype=float), 0.0, 1.0)
        if len(candidates) > len(rel):
            rel = np.pad(rel, (0, len(candidates) - len(rel)), constant_values=0.35)
        rel = rel[: len(candidates)]
        rng = np.random.default_rng(7)
        domains = sorted(corpus["domain"].dropna().unique().tolist())
        agent = EngagementRLAgent(arms=domains, policy="thompson")

        ranked = rerank(candidates, rel, agent, rng, RankConfig(), interest_text=interest)
        if len(ranked) < 10:
            _fail(report, name, f"Ranking returned only {len(ranked)} items")
            return

        top = ranked.head(10)
        if not live_mask(top).all():
            _fail(report, name, "Top-10 live-only results contain synthetic URLs")
            return

        rewards: list[float] = []
        actions = ["engage", "bookmark", "skip", "engage", "bookmark"]
        for i, (_, row) in enumerate(top.head(5).iterrows()):
            action = actions[i % len(actions)]
            r = agent.observe_feedback(str(row["domain"]), action)  # type: ignore[arg-type]
            rewards.append(r)
            headline = display_title(row)
            if not headline:
                _fail(report, name, f"Empty headline after action on id={row['id']}")
                return

        summary = agent.summary()
        if summary.get("rounds", 0) < 3:
            _fail(report, name, f"RL agent did not update enough: {summary}")
            return

        _pass(report, name)
    except Exception as exc:
        _fail(report, name, f"{exc}\n{traceback.format_exc()}")


def check_app_imports(report: Report) -> None:
    name = "app_imports"
    try:
        import app  # noqa: F401

        from engageiq.ui import render_opportunity_card  # noqa: F401

        _pass(report, name)
    except Exception as exc:
        _fail(report, name, str(exc))


def check_ui_native_components(report: Report) -> None:
    """Cards must use native Streamlit, not broken raw HTML."""
    name = "ui_native_cards"
    ui_path = CODE_DIR / "engageiq" / "ui.py"
    src = ui_path.read_text(encoding="utf-8")
    card_fn = src.split("def render_opportunity_card")[1].split("\ndef ")[0]
    if "unsafe_allow_html=True" in card_fn and "card-fact" not in src:
        _fail(report, name, "Card HTML must use card-fact chip styling")
        return
    if "st.container(border=True)" not in card_fn:
        _fail(report, name, "Cards should use st.container(border=True)")
        return
    if "display_title" not in card_fn:
        _fail(report, name, "Cards must use display_title() for real headlines")
        return
    _pass(report, name)


def check_suggest_action(report: Report, df: pd.DataFrame) -> None:
    name = "suggest_action_quality"
    from engageiq.suggestions import generate_suggestion

    live = df[live_mask(df)]
    seen: set[str] = set()
    for src in ("github", "gharchive"):
        sample = live[live["source"] == src].head(3)
        for _, row in sample.iterrows():
            suggestion = generate_suggestion(row, "machine learning developer tools")
            if len(suggestion) < 40:
                _fail(report, name, f"{src} id={row['id']}: suggestion too short")
                return
            if "looking for insights" in suggestion.lower():
                _fail(report, name, f"{src} id={row['id']}: generic suggestion")
                return
            sig = suggestion[:72]
            if sig in seen:
                _fail(report, name, f"{src} id={row['id']}: duplicate suggestion opener")
                return
            seen.add(sig)
    _pass(report, name)


def _check_multi_source_ingest(report: Report, df: pd.DataFrame) -> None:
    name = "multi_source_ingest"
    from engageiq.domains import DOMAINS

    sources = set(df["source"].dropna().astype(str).str.lower().unique())
    if not {"github", "gharchive"}.issubset(sources):
        _fail(report, name, f"Need GitHub API + GitHub Archive sources, got {sorted(sources)}")
        return
    present = set(df["domain"].dropna().astype(str).unique())
    missing = sorted(set(DOMAINS) - present)
    if missing:
        _fail(report, name, f"Missing required domains: {missing}")
        return
    _pass(report, name)


def _check_streaming_pipeline(report: Report, df: pd.DataFrame) -> None:
    name = "streaming_pipeline"
    try:
        from engageiq.streaming import OpportunityStream

        stream = OpportunityStream()
        batch = pd.DataFrame(
            [
                {"id": 1, "url": "https://example.local/a", "domain": "ML", "source": "github"},
                {"id": 2, "url": "https://example.local/a", "domain": "ML", "source": "github"},
            ]
        )
        stream.produce_rows(batch)
        if stream.pending() != 1:
            _fail(report, name, f"Expected 1 queued after dedup, got {stream.pending()}")
            return
        inserted, deduped = stream.consume(max_items=10, ingest_fn=lambda b: len(b))
        if inserted != 1 or deduped != 0:
            _fail(report, name, f"Expected ingest=1 dedup=0, got {inserted}/{deduped}")
            return
        _pass(report, name)
    except Exception as exc:
        _fail(report, name, str(exc))


def check_rl_improvement(report: Report, df: pd.DataFrame) -> None:
    name = "rl_improvement"
    from engageiq.persona_eval import PERSONAS, learning_benchmark

    lb = learning_benchmark(df, PERSONAS["Raj (Startup Founder / Marketing-Focused)"], rounds=60)
    if lb["rounds"] < 50:
        _fail(report, name, f"Only {lb['rounds']} rounds")
        return
    if lb["improvement"] <= 0 and lb["reward_improvement_last10"] <= 0:
        _fail(report, name, f"No measurable improvement: {lb}")
        return
    _pass(report, name)


def check_bundled_cloud_data(report: Report) -> None:
    name = "bundled_cloud_data"
    path = CODE_DIR / "data" / "live_opportunities.csv"
    if not path.exists():
        _fail(report, name, "code/data/live_opportunities.csv missing for Streamlit Cloud")
        return
    size_kb = path.stat().st_size / 1024
    if size_kb < 100:
        _fail(report, name, f"Bundled live CSV too small ({size_kb:.0f} KB)")
        return
    df = pd.read_csv(path)
    live = int(live_mask(df).sum())
    if live < 500:
        _fail(report, name, f"Bundled CSV has only {live} live rows")
        return
    _pass(report, name)


def run_all_checks() -> Report:
    report = Report()
    df, _store = load_dataframe()

    checks: list[Check] = [
        Check("submission_files", "package", lambda: check_submission_files(report)),
        Check("app_imports", "code", lambda: check_app_imports(report)),
        Check("ui_native_cards", "ui", lambda: check_ui_native_components(report)),
        Check("bundled_cloud_data", "deploy", lambda: check_bundled_cloud_data(report)),
        Check("dataset_size", "data", lambda: check_dataset_size(report, df)),
        Check("live_card_titles", "cards", lambda: check_live_card_titles(report, df)),
        Check("card_decision_info", "cards", lambda: check_card_decision_info(report, df)),
        Check("live_only_ranking", "ranking", lambda: check_live_only_ranking(report, df)),
        Check("plain_text_summaries", "cards", lambda: check_plain_text_summaries(report, df)),
        Check("persona_benchmarks", "benchmarks", lambda: check_persona_benchmarks(report, df)),
        Check("multi_source_ingest", "data", lambda: _check_multi_source_ingest(report, df)),
        Check("streaming_pipeline", "ux", lambda: _check_streaming_pipeline(report, df)),
        Check("user_session_simulation", "ux", lambda: simulate_user_session(report, df)),
        Check("rl_improvement", "benchmarks", lambda: check_rl_improvement(report, df)),
        Check("suggest_action_quality", "ux", lambda: check_suggest_action(report, df)),
    ]

    for chk in checks:
        if any(n == chk.name for n, _ in report.failed):
            continue
        try:
            chk.fn()
        except Exception as exc:
            _fail(report, chk.name, f"Unhandled: {exc}\n{traceback.format_exc()}")

    return report


def print_report(report: Report, round_num: int) -> None:
    print(f"\n{'='*60}")
    print(f"EngageIQ User Test — Round {round_num}")
    print(f"{'='*60}")
    print(f"PASSED: {len(report.passed)}  FAILED: {len(report.failed)}  WARNINGS: {len(report.warnings)}")
    if report.passed:
        print("\n[PASS] Passed:")
        for p in report.passed:
            print(f"  - {p}")
    if report.failed:
        print("\n[FAIL] Failed:")
        for name, err in report.failed:
            print(f"  - {name}: {err[:500]}{'...' if len(err) > 500 else ''}")
    if report.warnings:
        print("\n[WARN] Warnings:")
        for w in report.warnings:
            print(f"  - {w}")
    status = "ALL CHECKS PASSED" if report.ok else "CHECKS FAILED - fix and re-run"
    print(f"\n{status}")
    print(f"{'='*60}\n")


def refresh_artifacts() -> None:
    """Regenerate benchmarks, brief, and ZIP after fixes."""
    scripts = CODE_DIR / "scripts"
    for script in ("run_benchmarks.py", "generate_brief.py", "package_submission.py"):
        path = scripts / script
        if path.exists():
            print(f"Running {script}…")
            subprocess.run([sys.executable, str(path)], cwd=str(CODE_DIR), check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="EngageIQ user-test validation loop")
    parser.add_argument("--loop", action="store_true", help="Re-run until all pass or max rounds")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max loop iterations")
    parser.add_argument("--refresh", action="store_true", help="Regenerate benchmarks/brief/ZIP after pass")
    parser.add_argument("--json", type=str, default="", help="Write JSON report path")
    args = parser.parse_args()

    round_num = 0
    while True:
        round_num += 1
        report = run_all_checks()
        print_report(report, round_num)

        out_path = Path(args.json) if args.json else PROJECT_ROOT / "data" / "user_test_report.json"
        out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Report written to {out_path}")

        if report.ok:
            if args.refresh:
                refresh_artifacts()
            return 0

        if not args.loop or round_num >= args.max_rounds:
            return 1

        print(f"Waiting 2s before retry (round {round_num + 1}/{args.max_rounds})…")
        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
