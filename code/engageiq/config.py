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
    duckdb_path: Path


def _is_cloud_runtime() -> bool:
    """Detect Streamlit Cloud / other read-only container mounts."""
    if os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud":
        return True
    if os.getenv("STREAMLIT_CLOUD"):
        return True
    cwd = str(Path.cwd())
    if cwd.startswith("/mount"):
        return True
    return False


def _writable_duckdb_path(data_dir: Path) -> Path:
    """Streamlit Cloud repo is read-only — DuckDB must live in /tmp."""
    if _is_cloud_runtime():
        return Path(tempfile.gettempdir()) / "engageiq.duckdb"

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
    data_dir = project_root / "data"
    return Paths(
        code_dir=code_dir,
        project_root=project_root,
        data_dir=data_dir,
        snapshot_csv=data_dir / "opportunities_snapshot.csv",
        duckdb_path=_writable_duckdb_path(data_dir),
    )
