"""
Run user tests in a loop, refresh submission artifacts, and push to GitHub.

    py scripts/finalize_submission.py
    py scripts/finalize_submission.py --no-push
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = CODE_DIR / "scripts"


def run(cmd: list[str]) -> int:
    print(f"\n>> {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(CODE_DIR))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-push", action="store_true", help="Skip GitHub push")
    parser.add_argument("--max-rounds", type=int, default=5)
    args = parser.parse_args()

    code = run(
        [
            sys.executable,
            str(SCRIPTS / "user_test_loop.py"),
            "--loop",
            f"--max-rounds={args.max_rounds}",
            "--refresh",
        ]
    )
    if code != 0:
        print("\nUser tests failed. Fix issues and re-run finalize_submission.py")
        return code

    if args.no_push:
        print("\nAll tests passed. Skipping push (--no-push).")
        return 0

    push_code = run([sys.executable, str(SCRIPTS / "push_submission.py")])
    if push_code == 0:
        print("\nDone: tests passed, ZIP rebuilt, pushed to GitHub.")
        print("Streamlit Cloud will redeploy in ~2 minutes.")
    return push_code


if __name__ == "__main__":
    raise SystemExit(main())
