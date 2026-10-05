"""
Build a self-contained HTML viewer that compares the baseline and the Optuna-tuned
hyperparameters of the tree ensembles, one column of subplots per model.

Input is the json written by tune_trees.py (--out). The page has no external resources; the
results are embedded in it, so the output file can be opened or published as is.

Usage:
    python scripts/build_tuning_viewer.py results/tuning_trees.json -o results/tuning_viewer.html
"""

import argparse
import json

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src.utils.viewer_utils import build_data, render_viewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--test-start", default=None, help="default: from the tuning results")
    parser.add_argument("--tune-window", default=None)
    parser.add_argument("--test-window", default=None)
    args = parser.parse_args()

    data = build_data(json.loads(args.results.read_text()), args.test_start, args.tune_window, args.test_window)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_viewer(data), encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
