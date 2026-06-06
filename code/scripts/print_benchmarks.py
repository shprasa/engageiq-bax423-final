"""Print saved persona results from benchmark_results.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent
path = PROJECT_ROOT / "data" / "benchmark_results.json"
if len(sys.argv) > 1:
    path = Path(sys.argv[1])

data = json.loads(path.read_text(encoding="utf-8"))
ds = data["dataset"]
print("=== Saved benchmark_results.json ===")
print(f"Rows: {ds['dataset_rows']:,} | Domains: {ds['domains']}/15 | Synthetic: {ds['synthetic_rows']}")
print(f"Missing domains: {data['ingest_benchmark']['missing_domains']}")
print()
for row in data["personas"]:
    name = row["persona"].split("(")[0].strip()
    crit = ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in row["pass_criteria"].items())
    caps = sum(1 for v in row["capability_pass"].values() if v == "PASS")
    status = "PASS" if row["passed"] else "FAIL"
    print(f"{name}: {status} | criteria: {crit} | capabilities: {caps}/6")
lb = data["learning_benchmark"]
print(f"\nRL: {lb['rounds']} rounds, improvement={lb['improvement']:.3f}")
