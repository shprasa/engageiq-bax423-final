from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from engageiq.config import get_paths
from engageiq.data import OpportunityStore
from engageiq.persona_eval import PERSONAS, dataset_stats, evaluate_personas, learning_benchmark, sketch_benchmark


def main() -> None:
    paths = get_paths()
    store = OpportunityStore(paths.duckdb_path, snapshot_csv=paths.snapshot_csv)
    store.ensure_loaded_from_snapshot(paths.snapshot_csv, initial_ingest=100000)
    df = store.load_df()

    learning = learning_benchmark(df, PERSONAS["Raj (Startup Founder / Marketing-Focused)"], rounds=60)
    learning_ok = (
        learning["improvement"] > 0
        or learning["reward_improvement_last10"] > 0
        or learning.get("session_reward_gain", 0) > 0
    )
    persona_results = evaluate_personas(df, learning_ok=learning_ok)

    payload = {
        "dataset": dataset_stats(df),
        "sketch_benchmark": sketch_benchmark(df),
        "personas": [{**r.__dict__, "capability_pass": r.capability_pass} for r in persona_results],
        "learning_benchmark": learning,
    }
    out = paths.data_dir / "benchmark_results.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
