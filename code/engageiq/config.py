from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    code_dir: Path
    project_root: Path
    data_dir: Path
    snapshot_csv: Path
    live_csv: Path
    duckdb_path: Path
    is_cloud: bool


def _is_cloud_runtime(code_dir: Path) -> bool:
    root = str(code_dir).replace("\\", "/")
    if root.startswith("/mount"):
        return True
    if os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud":
        return True
    if os.getenv("STREAMLIT_CLOUD"):
        return True
    return False


def _resolve_snapshot_csv(code_dir: Path, project_root: Path) -> Path:
    """Find the best available snapshot on local disk or Streamlit Cloud."""
    candidates = [
        project_root / "data" / "opportunities_snapshot.csv",
        code_dir / "data" / "opportunities_snapshot.csv",
        code_dir / "data" / "live_opportunities.csv",
    ]
    best: Path | None = None
    best_score = (-1, -1)  # (live_rows_hint, file_size)
    for path in candidates:
        if not path.exists():
            continue
        size = path.stat().st_size
        # Prefer full snapshot over live-only when both exist
        score = (2 if "opportunities_snapshot" in path.name else 1, size)
        if score > best_score:
            best_score = score
            best = path
    if best is None:
        return project_root / "data" / "opportunities_snapshot.csv"
    return best


def _resolve_live_csv(code_dir: Path, project_root: Path) -> Path:
    candidates = [
        code_dir / "data" / "live_opportunities.csv",
        project_root / "data" / "live_opportunities.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return code_dir / "data" / "live_opportunities.csv"


def _duckdb_path(data_dir: Path, is_cloud: bool) -> Path:
    if is_cloud:
        return Path("/tmp/engageiq.duckdb")
    target = data_dir / "engageiq.duckdb"
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return target
    except OSError:
        return Path(tempfile.gettempdir()) / "engageiq.duckdb"


def get_paths() -> Paths:
    code_dir = Path(__file__).resolve().parent.parent
    project_root = code_dir.parent
    is_cloud = _is_cloud_runtime(code_dir)

    # On Streamlit Cloud the repo root is often the parent of `code/`.
    data_dir = project_root / "data"
    if is_cloud and not (data_dir / "opportunities_snapshot.csv").exists():
        data_dir = code_dir / "data"

    snapshot_csv = _resolve_snapshot_csv(code_dir, project_root)
    live_csv = _resolve_live_csv(code_dir, project_root)

    return Paths(
        code_dir=code_dir,
        project_root=project_root,
        data_dir=data_dir,
        snapshot_csv=snapshot_csv,
        live_csv=live_csv,
        duckdb_path=_duckdb_path(data_dir, is_cloud),
        is_cloud=is_cloud,
    )
