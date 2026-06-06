"""Build Prasad_Shivneel_BAX423_Final.zip for Canvas submission."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent
SUBMISSION_DIR = PROJECT_ROOT.parent / "EngageIQ_Final_Submission"

ZIP_NAME = "Prasad_Shivneel_BAX423_Final.zip"

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".cursor", "_tmp_engageiq_docx"}
SKIP_FILES = {
    ".env",
    "engageiq.duckdb",
    "engageiq.duckdb.wal",
    "Thumbs.db",
    ".DS_Store",
    "push_log.txt",
    "build_snapshot.log",
    "build_github_topup.log",
}
SKIP_SUFFIX = {".pyc", ".duckdb", ".wal"}


def _should_include(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in SKIP_DIRS for part in rel.parts):
        return False
    if path.name in SKIP_FILES:
        return False
    if path.suffix.lower() in SKIP_SUFFIX:
        return False
    return True


def _ensure_brief_pdf() -> Path:
    pdf = PROJECT_ROOT / "brief.pdf"
    if not pdf.exists():
        raise FileNotFoundError(
            f"Missing {pdf}. Convert brief.docx to brief.pdf and place it in the project root."
        )
    return pdf


def _collect_files() -> list[tuple[Path, str]]:
    include_roots = [
        (PROJECT_ROOT / "code", "code"),
        (PROJECT_ROOT / "data", "data"),
        (PROJECT_ROOT / "brief.pdf", "brief.pdf"),
        (PROJECT_ROOT / "prompts.md", "prompts.md"),
    ]
    files: list[tuple[Path, str]] = []
    for root, arc_prefix in include_roots:
        if root.is_file():
            if _should_include(root, PROJECT_ROOT):
                files.append((root, arc_prefix))
        elif root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file() and _should_include(path, root):
                    arc = f"{arc_prefix}/{path.relative_to(root).as_posix()}"
                    files.append((path, arc))
    return files


def build_zip(out_path: Path | None = None) -> Path:
    _ensure_brief_pdf()
    out = out_path or SUBMISSION_DIR / ZIP_NAME
    files = _collect_files()

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, arc in files:
            zf.write(path, arcname=arc)

    print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(files)} files)")
    return out


def verify_local_run(zip_path: Path) -> None:
    """Unzip and confirm the grader run path from brief.docx works offline."""
    with tempfile.TemporaryDirectory(prefix="engageiq_verify_") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)

        required = [
            root / "code" / "app.py",
            root / "code" / "requirements.txt",
            root / "data" / "opportunities_snapshot.csv",
            root / "brief.pdf",
            root / "prompts.md",
        ]
        missing = [str(p.relative_to(root)) for p in required if not p.exists()]
        if missing:
            raise RuntimeError(f"ZIP missing required paths: {missing}")

        env = {**os.environ, "PYTHONPATH": str(root / "code")}
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
            cwd=str(root / "code"),
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from engageiq.config import get_paths; "
                "from engageiq.data import OpportunityStore; "
                "p=get_paths(); "
                "assert p.snapshot_csv.exists(), p.snapshot_csv; "
                "s=OpportunityStore(p.duckdb_path, snapshot_csv=p.snapshot_csv); "
                "s.ensure_loaded_from_snapshot(p.snapshot_csv, initial_ingest=100000); "
                "df=s.load_df(); "
                "assert len(df) >= 10000, len(df); "
                "print(f'OK: {len(df)} rows from {p.snapshot_csv}')",
            ],
            cwd=str(root / "code"),
            check=True,
            env=env,
        )
        print("Verified: cd code && pip install && offline snapshot load (brief local run path)")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    zip_path = build_zip(args.output)
    if not args.no_verify:
        verify_local_run(zip_path)


if __name__ == "__main__":
    main()
