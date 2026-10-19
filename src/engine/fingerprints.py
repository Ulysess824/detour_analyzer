"""Short, stable fingerprints of the data, the parameters and the files a result depends on."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.engine.panel_utils import Panel


def panel_fingerprint(S: Panel) -> str:
    r"""
    Short hash of the data of a panel: the series (in order), the months and the monthly totals.

    It is computed from the panel and not from the csv bytes, so a different line ending does not
    change it.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps([list(k) for k in S.keys]).encode())
    digest.update(json.dumps(S.labels).encode())
    digest.update(np.round(np.nan_to_num(S.monthly, nan=-1.0), 6).astype("<f8").tobytes())
    return digest.hexdigest()[:16]


def params_hash(tuned: dict) -> str:
    r"""Short hash of the tuned hyperparameters that the member forecasts depend on."""
    best = {kind: result["best_params"] for kind, result in sorted(tuned.items())}
    return hashlib.sha256(json.dumps(best, sort_keys=True).encode()).hexdigest()[:16]


def file_sha256(path: str | Path) -> str:
    r"""SHA-256 of the bytes of a file (read in blocks)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
