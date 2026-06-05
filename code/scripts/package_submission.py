from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESKTOP = ROOT.parent
ZIP_NAME = "Prasad_Shivneel_BAX423_Final.zip"
STAGING = DESKTOP / "_submission_staging"
SKIP = {".env", "__pycache__", ".git", ".venv", "venv"}


def should_skip(path: Path) -> bool:
    if path.name in SKIP:
        return True
    if path.suffix in {".duckdb", ".duckdb.wal", ".pyc"}:
        return True
    return any(p in SKIP for p in path.parts)


def main() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    (STAGING / "code").mkdir(parents=True)
    (STAGING / "data").mkdir(parents=True)

    for src, dst_name in [
        (ROOT / "code", STAGING / "code"),
        (ROOT / "data" / "opportunities_snapshot.csv", STAGING / "data" / "opportunities_snapshot.csv"),
        (ROOT / "data" / "benchmark_results.json", STAGING / "data" / "benchmark_results.json"),
        (ROOT / "brief.pdf", STAGING / "brief.pdf"),
        (ROOT / "prompts.md", STAGING / "prompts.md"),
    ]:
        if src.is_dir():
            for p in src.rglob("*"):
                if p.is_dir() or should_skip(p):
                    continue
                rel = p.relative_to(src)
                out = STAGING / "code" / rel if dst_name.name == "code" else dst_name
                if dst_name.name == "code":
                    out = STAGING / "code" / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, out)
        elif src.exists():
            dst_name.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_name)

    zip_path = DESKTOP / ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in STAGING.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(STAGING).as_posix())

    shutil.rmtree(STAGING)
    print(f"Created {zip_path}")


if __name__ == "__main__":
    main()
