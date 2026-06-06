"""Quick persona pass/fail check (main repo or extracted submission folder)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from engageiq.config import get_paths
from engageiq.data import OpportunityStore
from engageiq.domains import DOMAINS
from engageiq.persona_eval import PERSONAS, evaluate_personas, learning_benchmark


def main() -> int:
    paths = get_paths()
    print(f"Project root: {paths.project_root}")
    print(f"Snapshot CSV: {paths.snapshot_csv} (exists={paths.snapshot_csv.exists()})")

    store = OpportunityStore(paths.duckdb_path, snapshot_csv=paths.snapshot_csv)
    store.ensure_loaded_from_snapshot(paths.snapshot_csv, initial_ingest=100000)
    df = store.load_df()

    present = sorted(df["domain"].dropna().astype(str).unique())
    missing = sorted(set(DOMAINS) - set(present))
    print(f"Dataset: {len(df):,} rows, {len(present)}/15 domains")
    if missing:
        print(f"MISSING DOMAINS: {missing}")
        return 1

    learning = learning_benchmark(df, PERSONAS["Raj (Startup Founder / Marketing-Focused)"], rounds=60)
    learning_ok = learning["improvement"] > 0 or learning["reward_improvement_last10"] > 0
    results = evaluate_personas(df, learning_ok=learning_ok)

    all_ok = True
    print("\n=== Persona criteria ===")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"\n{r.persona}: {status}")
        for k, v in (r.pass_criteria or {}).items():
            mark = "ok" if v else "FAIL"
            print(f"  - {k}: {mark}")
        if not r.passed:
            all_ok = False

    print("\n=== Core capabilities (all personas) ===")
    for r in results:
        partial = [k for k, v in r.capability_pass.items() if v != "PASS"]
        if partial:
            print(f"{r.persona}: {partial}")
            all_ok = False
        else:
            print(f"{r.persona}: 6/6 PASS")

    print(f"\nRL: {learning['rounds']} rounds, improvement={learning['improvement']:.3f}")
    out = paths.data_dir / "benchmark_results.json"
    print(f"\nOverall: {'ALL PASS' if all_ok else 'FAILURES DETECTED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
