from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DESKTOP = PROJECT_ROOT.parent
ZIP_NAME = "Prasad_Shivneel_BAX423_Final.zip"
STAGING = DESKTOP / "_submission_staging"
SKIP = {".env", "__pycache__", ".git", ".venv", "venv", ".streamlit/deploy_trigger.txt"}


def should_skip(path: Path) -> bool:
    if path.name in SKIP:
        return True
    if path.suffix in {".duckdb", ".duckdb.wal", ".pyc"}:
        return True
    return any(part in SKIP for part in path.parts)


def main() -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    (STAGING / "code").mkdir(parents=True)
    (STAGING / "data").mkdir(parents=True)

    code_src = PROJECT_ROOT / "code"
    for p in code_src.rglob("*"):
        if p.is_dir() or should_skip(p):
            continue
        rel = p.relative_to(code_src)
        out = STAGING / "code" / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)

    for src in [
        PROJECT_ROOT / "data" / "opportunities_snapshot.csv",
        PROJECT_ROOT / "data" / "benchmark_results.json",
        PROJECT_ROOT / "brief.pdf",
        PROJECT_ROOT / "prompts.md",
    ]:
        if src.exists():
            dst = STAGING / src.relative_to(PROJECT_ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

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
