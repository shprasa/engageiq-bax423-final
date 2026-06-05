"""Build Prasad_Shivneel_BAX423_Final.zip for Canvas submission."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent

ZIP_NAME = "Prasad_Shivneel_BAX423_Final.zip"

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".cursor", "_tmp_engageiq_docx"}
SKIP_FILES = {".env", "engageiq.duckdb", "engageiq.duckdb.wal", "Thumbs.db", ".DS_Store"}
SKIP_SUFFIX = {".pyc", ".duckdb", ".wal"}


def _should_include(path: Path) -> bool:
    rel = path.relative_to(PROJECT_ROOT)
    if any(part in SKIP_DIRS for part in rel.parts):
        return False
    if path.name in SKIP_FILES:
        return False
    if path.suffix.lower() in SKIP_SUFFIX:
        return False
    return True


def build_zip(out_path: Path | None = None) -> Path:
    out = out_path or PROJECT_ROOT / ZIP_NAME
    include_roots = [
        PROJECT_ROOT / "code",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "brief.pdf",
        PROJECT_ROOT / "prompts.md",
        PROJECT_ROOT / "README.md",
    ]
    files: list[Path] = []
    for root in include_roots:
        if root.is_file():
            if _should_include(root):
                files.append(root)
        elif root.is_dir():
            for p in sorted(root.rglob("*")):
                if p.is_file() and _should_include(p):
                    files.append(p)

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            arc = path.relative_to(PROJECT_ROOT).as_posix()
            zf.write(path, arcname=arc)

    print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(files)} files)")
    return out


def main() -> None:
    build_zip()


if __name__ == "__main__":
    main()
