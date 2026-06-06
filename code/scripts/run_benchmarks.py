from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from engageiq.benchmark_ops import run_and_write_benchmarks
from engageiq.config import get_paths
from engageiq.data import OpportunityStore


def main() -> None:
    paths = get_paths()
    store = OpportunityStore(paths.duckdb_path, snapshot_csv=paths.snapshot_csv)
    store.ensure_loaded_from_snapshot(paths.snapshot_csv, initial_ingest=100000)
    df = store.load_df()

    payload = run_and_write_benchmarks(df, paths)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {paths.data_dir / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
