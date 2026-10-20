"""
Start the desktop app from the source code (the packaged version is detour.exe).

Usage:
    python scripts/run_app.py             # window (needs pywebview) or, without it, the browser
    python scripts/run_app.py --browser   # always the browser
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.app.launcher import main

if __name__ == "__main__":
    raise SystemExit(main())
