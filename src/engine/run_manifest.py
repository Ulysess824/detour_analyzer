"""
Run manifest: what a forecast was made with, written next to it (`forecast_<month>.manifest.json`).

It records the code (commit, branch, uncommitted changes), the data (file hash, rows, date range, series, panel fingerprint, summary of the
input checks), the parameters (hash, tuning windows, seed), the environment (library versions) and the output (rows, totals, hash of the csv).
The data itself is not copied: the commit plus the file hash identify it when the file is in git.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime
from importlib import metadata
from pathlib import Path

import pandas as pd

from src.engine.data_checks import Report
from src.engine.fingerprints import file_sha256, panel_fingerprint, params_hash
from src.engine.member_utils import SEED
from src.engine.panel_utils import Panel

SCHEMA = 1
PACKAGES = ["pandas", "numpy", "lightgbm", "xgboost", "scikit-learn"]


def _git(repo_dir: str | Path, *args: str) -> str | None:
    r"""Output of a git command run in `repo_dir`, or None when git is missing, the directory is not a repository or the command fails."""
    try:
        done = subprocess.run(["git", "-C", str(repo_dir), *args], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_info(repo_dir: str | Path) -> dict:
    r"""Commit, branch and whether tracked files have uncommitted changes; 'unknown' and None when it cannot be told (never fails)."""
    commit = _git(repo_dir, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": "unknown", "branch": "unknown", "dirty": None}
    status = _git(repo_dir, "status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "branch": _git(repo_dir, "rev-parse", "--abbrev-ref", "HEAD") or "unknown", "dirty": None if status is None else bool(status)}


def file_git_state(path: str | Path) -> dict:
    r"""Whether a file is tracked by git and whether it differs from the last commit (None when unknown)."""
    path = Path(path)
    folder = path.parent if path.parent.exists() else Path(".")
    tracked = _git(folder, "ls-files", "--error-unmatch", path.name)
    if _git(folder, "rev-parse", "HEAD") is None:
        return {"in_git": None, "modified": None}
    if tracked is None:
        return {"in_git": False, "modified": None}
    status = _git(folder, "status", "--porcelain", "--", path.name)
    return {"in_git": True, "modified": None if status is None else bool(status)}


def environment_info() -> dict:
    r"""Python, operating system and the versions of the libraries that decide the numbers."""
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for package in PACKAGES:
        try:
            info[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            info[package] = "not installed"
    return info


def manifest_path(forecast_path: str | Path) -> Path:
    r"""`results/forecast_2026-10.csv` -> `results/forecast_2026-10.manifest.json`."""
    return Path(forecast_path).with_suffix(".manifest.json")


def write_manifest(path: str | Path, manifest: dict) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_manifest(
    *,
    S: Panel,
    df: pd.DataFrame,
    input_path: str | Path,
    tuned: dict,
    tuning_path: str | Path | None,
    result: pd.DataFrame,
    forecast_path: str | Path,
    as_of: str | None,
    started_at: datetime,
    seconds: float,
    git: dict,
    validation: Report | None = None,
    rows_in_file: int | None = None,
    strategy: dict | None = None,
) -> dict:
    r"""
    The manifest of one forecast run.

    `df` is the consumption table the panel `S` was built from (after `--as-of`), `result` the forecast table, `forecast_path` the csv already
    written, `git` the output of `git_info` taken before the run (so the files the run writes do not count as changes).
    """
    input_path, forecast_path = Path(input_path), Path(forecast_path)
    fingerprint, hash_params = panel_fingerprint(S), params_hash(tuned) if tuned else "none"
    created = started_at.isoformat(timespec="seconds")
    short = hashlib.sha256(f"{created}|{fingerprint}|{hash_params}|{git['commit']}".encode()).hexdigest()[:6]
    warnings = {}
    if validation is not None:
        warnings = {f.code: (len(f.rows) if f.rows is not None else 1) for f in validation.warnings}
    exists = input_path.exists()
    state = file_git_state(input_path) if exists else {"in_git": None, "modified": None}
    data = {
        "path": str(input_path),
        "file_sha256": file_sha256(input_path) if exists else None,
        "file_in_git": state["in_git"],
        "file_modified": state["modified"],
        "rows": int(len(df)),
        "first_date": str(df["fecha"].min().date()),
        "last_date": str(df["fecha"].max().date()),
        "n_series": int(len(S.keys)),
        "panel_fingerprint": fingerprint,
        "validation": {"errors": 0 if validation is None else len(validation.errors), "warnings": warnings},
    }
    if rows_in_file is not None:
        data["rows_in_file"] = int(rows_in_file)
    tuning_exists = tuning_path is not None and Path(tuning_path).exists()
    params = {
        "tuning_file": None if tuning_path is None else str(tuning_path),
        "tuning_file_sha256": file_sha256(tuning_path) if tuning_exists else None,
        "params_hash": hash_params,
        "tune_window": {kind: r.get("meta", {}).get("tune_window") for kind, r in sorted(tuned.items())},
        "test_window": {kind: r.get("meta", {}).get("test_window") for kind, r in sorted(tuned.items())},
        "seed": SEED,
    }
    by_planta = result.groupby("planta")["ml_mean"].sum().round(3)
    output = {
        "csv": str(forecast_path),
        "csv_sha256": file_sha256(forecast_path),
        "rows": int(len(result)),
        "total_ml_mean": round(float(result["ml_mean"].sum()), 3),
        "by_planta": {planta: float(value) for planta, value in by_planta.items()},
    }
    return {
        "schema": SCHEMA,
        "run": {
            "run_id": f"{started_at:%Y%m%dT%H%M%SZ}-{short}",
            "created_at": created,
            "forecast_month": str(result["mes"].iloc[0]),
            "as_of": as_of,
            "seconds": round(float(seconds), 2),
        },
        "code": git,
        "data": data,
        "params": params,
        "environment": environment_info(),
        "output": output,
        **({"strategy": strategy} if strategy is not None else {}),
    }
