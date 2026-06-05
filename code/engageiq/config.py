from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    code_dir: Path
    project_root: Path
    data_dir: Path
    snapshot_csv: Path
    duckdb_path: Path


def get_paths() -> Paths:
    code_dir = Path(__file__).resolve().parent.parent
    project_root = code_dir.parent
    data_dir = project_root / "data"
    return Paths(
        code_dir=code_dir,
        project_root=project_root,
        data_dir=data_dir,
        snapshot_csv=data_dir / "opportunities_snapshot.csv",
        duckdb_path=data_dir / "engageiq.duckdb",
    )

