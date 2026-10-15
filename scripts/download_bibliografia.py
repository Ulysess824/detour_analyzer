"""
Download the papers of the bibliografia/ folder (open-access copies only).

The PDFs are not tracked by git (bibliografia/.gitignore): this script rebuilds them. A paper that cannot be
downloaded (for example because the network policy blocks the host) is reported and the others continue.

Usage:
    python scripts/download_bibliografia.py
    python scripts/download_bibliografia.py --list
    python scripts/download_bibliografia.py --only giannone2008 --overwrite
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.bibliography_utils import REFERENCES, download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=Path("bibliografia"))
    parser.add_argument("--only", nargs="+", default=None, help="keys of the references to download (default: all)")
    parser.add_argument("--list", action="store_true", help="show the references and exit")
    parser.add_argument("--overwrite", action="store_true", help="download again even if the file exists")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    chosen = [r for r in REFERENCES if args.only is None or r.key in args.only]
    unknown = set(args.only or []) - {r.key for r in REFERENCES}
    if unknown:
        raise SystemExit(f"Unknown keys: {sorted(unknown)}. Available: {[r.key for r in REFERENCES]}")
    if args.list:
        for r in chosen:
            print(f"{r.key}: {r.citation}\n    file {r.filename}\n    " + "\n    ".join(r.urls))
        return

    failed = 0
    for r in chosen:
        status, detail = download(r, args.out_dir, args.timeout, args.overwrite)
        print(f"[{status}] {r.key} -> {args.out_dir / r.filename}\n    {detail}")
        failed += status == "failed"
    if failed:
        raise SystemExit(f"{failed} of {len(chosen)} references could not be downloaded.")


if __name__ == "__main__":
    main()
