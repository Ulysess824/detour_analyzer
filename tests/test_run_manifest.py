"""Run manifest: what was used to make a forecast is written next to it, and it changes exactly when something used changes."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.engine.data_checks import check_consumption
from src.engine.fingerprints import file_sha256, panel_fingerprint, params_hash
from src.engine.inference_utils import forecast_next_month
from src.engine.panel_utils import build_panel
from src.engine.run_manifest import build_manifest, environment_info, git_info, manifest_path, write_manifest

REPO = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 10, 19, 12, 0, 0, tzinfo=timezone.utc)
TUNED = {"lgbm": {"best_params": {"learning_rate": 0.03, "num_leaves": 8}, "meta": {"tune_window": "2024-01 a 2025-09", "test_window": "2025-10 a 2026-09"}}}


def make(df, tmp_path, tuned=None, name="forecast.csv", as_of=None, started=T0, seconds=1.0):
    S = build_panel(df)
    result = forecast_next_month(S, tuned or {})
    csv = tmp_path / name
    result.round(3).to_csv(csv, index=False)
    manifest = build_manifest(
        S=S, df=df, input_path=tmp_path / "in.csv", tuned=tuned or {}, tuning_path=None, result=result, forecast_path=csv,
        as_of=as_of, started_at=started, seconds=seconds, git=git_info(REPO), validation=check_consumption(df),
    )
    return manifest, S, result, csv


def test_the_manifest_has_every_block_and_is_json(toy_df, tmp_path):
    manifest, S, result, csv = make(toy_df, tmp_path)
    assert set(manifest) == {"schema", "run", "code", "data", "params", "environment", "output"}
    assert set(manifest["run"]) == {"run_id", "created_at", "forecast_month", "as_of", "seconds"}
    assert manifest["run"]["forecast_month"] == result["mes"].iloc[0] and manifest["run"]["created_at"].startswith("2026-10-19T12:00:00")
    assert manifest["data"]["panel_fingerprint"] == panel_fingerprint(S) and manifest["data"]["n_series"] == len(S.keys)
    assert manifest["data"]["rows"] == len(toy_df) and manifest["data"]["first_date"] == str(toy_df["fecha"].min().date())
    assert manifest["output"]["csv_sha256"] == file_sha256(csv) and manifest["output"]["rows"] == len(result)
    assert np.isclose(manifest["output"]["total_ml_mean"], result["ml_mean"].sum(), atol=1e-3)
    assert np.isclose(sum(manifest["output"]["by_planta"].values()), result["ml_mean"].sum(), atol=1e-2)
    assert set(manifest["code"]) == {"commit", "branch", "dirty"} and set(manifest["environment"]) >= {"python", "pandas", "numpy", "lightgbm", "xgboost", "scikit-learn"}
    path = manifest_path(csv)
    write_manifest(path, manifest)
    assert path.name == "forecast.manifest.json" and json.loads(path.read_text()) == manifest


def test_the_same_inputs_give_the_same_fingerprints_and_the_same_output_hash(toy_df, tmp_path):
    a, *_ = make(toy_df, tmp_path, name="a.csv", started=T0)
    b, *_ = make(toy_df, tmp_path, name="b.csv", started=datetime(2026, 11, 2, 9, 30, tzinfo=timezone.utc), seconds=9.0)
    assert a["data"] == b["data"] and a["params"] == b["params"] and a["environment"] == b["environment"]
    assert a["output"]["csv_sha256"] == b["output"]["csv_sha256"] and a["output"]["by_planta"] == b["output"]["by_planta"]
    assert a["run"]["run_id"] != b["run"]["run_id"] and a["run"]["created_at"] != b["run"]["created_at"]


def test_changing_the_data_or_the_parameters_changes_the_fingerprints(toy_df, tmp_path):
    base, *_ = make(toy_df, tmp_path, name="base.csv")
    other = toy_df.copy()
    other.loc[other.index[-5:], "consumo"] += 3.0
    changed_data, *_ = make(other, tmp_path, name="d.csv")
    changed_params, *_ = make(toy_df, tmp_path, tuned=TUNED, name="p.csv")
    assert changed_data["data"]["panel_fingerprint"] != base["data"]["panel_fingerprint"]
    assert changed_params["params"]["params_hash"] != base["params"]["params_hash"] == "none"
    assert changed_params["params"]["params_hash"] == params_hash(TUNED)
    assert changed_params["params"]["tune_window"] == {"lgbm": "2024-01 a 2025-09"}


def test_as_of_and_the_validation_summary_are_recorded(toy_df, tmp_path):
    manifest, *_ = make(toy_df, tmp_path, as_of="2025-05")
    assert manifest["run"]["as_of"] == "2025-05"
    assert manifest["data"]["validation"]["errors"] == 0 and isinstance(manifest["data"]["validation"]["warnings"], dict)


def test_git_info_inside_the_repo_and_outside_it(tmp_path):
    inside = git_info(REPO)
    assert inside["commit"] == "unknown" or (len(inside["commit"]) >= 7 and isinstance(inside["dirty"], bool))
    outside = git_info(tmp_path)  # not a git repository: the run must not fail
    assert outside == {"commit": "unknown", "branch": "unknown", "dirty": None}


def test_environment_lists_versions():
    env = environment_info()
    assert env["python"].count(".") >= 1 and all(isinstance(v, str) for v in env.values())


def test_file_sha256_is_the_content_hash(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"abc")
    assert file_sha256(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_predict_next_month_writes_the_manifest_next_to_the_forecast(toy_df, tmp_path):
    data, out = tmp_path / "consumos.csv", tmp_path / "forecast_x.csv"
    toy_df.to_csv(data, index=False)
    run = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "predict_next_month.py"), str(data), "--out", str(out), "--tuning", str(tmp_path / "none.json")],
        capture_output=True, text=True, cwd=REPO,
    )
    assert run.returncode == 0, run.stderr[-500:]
    manifest = json.loads(manifest_path(out).read_text())
    assert manifest["output"]["csv_sha256"] == file_sha256(out) and manifest["data"]["file_sha256"] == file_sha256(data)
    assert manifest["run"]["seconds"] > 0 and manifest["data"]["path"] == str(data)
