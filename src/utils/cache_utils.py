"""
Guards for results that are stored on disk and reused later.

The cached member forecasts refer to a series by its position in the alphabetical list of
(planta, sku) pairs. If the master csv gains or loses a SKU, every position after it shifts and the
cache silently describes other series. These functions detect that, and a tuning that overlaps the
test months (which would leak the test into the chosen hyperparameters).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.panel_utils import Panel


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


def expected_meta(S: Panel, horizon: int, n_test: int, tuned: dict) -> dict:
    r"""What a cache of member forecasts must say about itself to be valid for this data and settings."""
    n_months = S.monthly.shape[1]
    return {
        "data": panel_fingerprint(S),
        "n_series": len(S.keys),
        "horizon": horizon,
        "test_first": S.labels[n_months - n_test],
        "test_last": S.labels[-1],
        "tuning": params_hash(tuned) if tuned else "none",
    }


def meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".meta.json")


def write_meta(cache_path: Path, meta: dict) -> None:
    meta_path(cache_path).write_text(json.dumps(meta, indent=1))


def check_meta(cache_path: Path, expected: dict, keys: tuple[str, ...] | None = None) -> None:
    r"""Stop with a clear message if the cache was made for other data or settings (compare only `keys` if given)."""
    path = meta_path(cache_path)
    if not path.exists():
        raise SystemExit(f"{cache_path} has no {path.name}, so it cannot be trusted. Recompute it (--refit).")
    stored = json.loads(path.read_text())
    wanted = keys or tuple(expected)
    different = [f"{k}: stored {stored.get(k)!r}, now {expected[k]!r}" for k in wanted if stored.get(k) != expected[k]]
    if different:
        raise SystemExit(f"{cache_path} is out of date ({'; '.join(different)}). Recompute it (--refit).")


def check_alignment(res: pd.DataFrame, S: Panel) -> None:
    r"""
    Check, from the content, that the series and months of `res` match the panel.

    Needs the columns series, t, y and planta_code. This catches a shifted series index even
    when no metadata exists.
    """
    series, month = res["series"].to_numpy(), res["t"].to_numpy()
    if series.max() >= len(S.keys) or month.max() >= S.monthly.shape[1]:
        raise SystemExit("The forecasts refer to series or months that the current data does not have.")
    if not np.allclose(S.monthly[series, month], res["y"].to_numpy(), atol=1e-6):
        raise SystemExit("The actuals stored with the forecasts differ from the current data (series index shifted?).")
    if not np.array_equal(S.planta_code[series], res["planta_code"].to_numpy()):
        raise SystemExit("The plantas stored with the forecasts differ from the current data (series index shifted?).")
