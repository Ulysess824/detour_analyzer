"""The runs the command line scripts and the desktop app share: forecast, tracking and tuning."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.engine.feature_utils import make_frame
from src.engine.fingerprints import file_sha256, panel_fingerprint
from src.engine.inference_utils import forecast_next_month
from src.engine.member_utils import collect_member_forecasts
from src.engine.panel_utils import build_panel
from src.engine.prediction_run import run_prediction
from src.evaluation.cache_utils import write_meta
from src.evaluation.tracking_run import BACKTEST_RUN, save_tracking, track_backtest, track_saved_forecast
from src.evaluation.tuning_run import run_tuning


@pytest.fixture()
def data(toy_df, tmp_path):
    path = tmp_path / "consumos.csv"
    toy_df.to_csv(path, index=False)
    return path


def test_a_forecast_run_saves_the_csv_and_the_manifest_and_reports_progress(data, toy_panel, tmp_path):
    steps = []
    run = run_prediction(data, tmp_path / "none.json", tmp_path / "forecast_x.csv", progress=lambda message, fraction: steps.append((message, fraction)))
    fractions = [f for _, f in steps]
    assert fractions == sorted(fractions) and fractions[0] < 0.2 and fractions[-1] == 1.0 and all(isinstance(m, str) and m for m, _ in steps)
    assert run.forecast_path.exists() and run.manifest_file.exists() and json.loads(run.manifest_file.read_text())["run"]["run_id"] == run.manifest["run"]["run_id"]
    assert run.manifest["output"]["csv_sha256"] == file_sha256(run.forecast_path) and run.manifest["data"]["panel_fingerprint"] == panel_fingerprint(toy_panel)
    expected = forecast_next_month(toy_panel, {})
    assert run.month == expected["mes"].iloc[0] and np.allclose(run.result["ml_mean"], expected["ml_mean"])
    assert pd.read_csv(run.forecast_path)["ml_mean"].round(3).equals(expected["ml_mean"].round(3))


def _partial(toy_df, tmp_path, last="2025-06-20"):
    path = tmp_path / "partial.csv"
    toy_df[toy_df["fecha"] <= last].to_csv(path, index=False)
    return path


def test_a_forecast_run_with_the_month_in_progress_forecasts_the_month_after_it(toy_df, tmp_path):
    run = run_prediction(_partial(toy_df, tmp_path), None, tmp_path / "f.csv")
    assert run.state.in_progress and run.state.cutoff_day == 20
    assert (run.plan.strategy, run.plan.cutoff_day, run.plan.origin, run.plan.target) == ("parcial", 20, "2025-05", "2025-07")
    assert run.month == "2025-07" and list(run.result.columns) == ["planta", "sku", "mes", "lgbm", "xgb", "rf", "ml_mean", "naive"]
    record = run.manifest["strategy"]
    assert (record["requested"], record["used"], record["cutoff_day"], record["target_month"]) == ("automatica", "parcial", 20, "2025-07")
    assert record["estimated_months"] == [] and record["month_state"]["status"] == "in_progress" and record["month_state"]["last_date"] <= "2025-06-20"


def test_a_forecast_that_is_not_one_month_ahead_does_not_take_the_name_of_the_one_month_forecast(toy_df, tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    data = _partial(toy_df, tmp_path)
    monkeypatch.chdir(tmp_path)
    run = run_prediction(data, None, repo_dir=repo)
    assert run.forecast_path.name == "forecast_2025-07_parcial.csv" and run.forecast_path.exists()
    assert run_prediction(data, None, strategy="h1", repo_dir=repo).forecast_path.name == "forecast_2025-06.csv"


def test_the_strategy_and_the_cutoff_can_be_chosen(toy_df, tmp_path):
    data = _partial(toy_df, tmp_path)
    run = run_prediction(data, None, tmp_path / "n.csv", strategy="nowcast_media", cutoff_day=12)
    assert (run.plan.strategy, run.plan.cutoff_day) == ("nowcast_media", 12) and run.manifest["strategy"]["estimated_months"] == ["2025-06"]
    short = run_prediction(_partial(toy_df, tmp_path, "2025-06-06"), None, tmp_path / "d.csv")
    assert short.plan.strategy == "directa" and short.plan.notes  # too few days of the month in progress
    with pytest.raises(SystemExit, match="between 1 and"):
        run_prediction(data, None, tmp_path / "x.csv", cutoff_day=28)
    with pytest.raises(SystemExit, match="unknown strategy"):
        run_prediction(data, None, tmp_path / "x.csv", strategy="otra")


def test_a_forecast_run_as_of_the_complete_month_before_ignores_the_month_in_progress(toy_df, tmp_path):
    run = run_prediction(_partial(toy_df, tmp_path), None, tmp_path / "f.csv", as_of="2025-05")
    assert run.month == "2025-06" and run.manifest["run"]["as_of"] == "2025-05"
    assert run.plan.strategy == "h1" and not run.state.in_progress and run.manifest["strategy"]["used"] == "h1"


def test_a_saved_forecast_is_tracked_against_the_real_and_stored(data, toy_df, toy_panel, tmp_path):
    run = run_prediction(data, None, tmp_path / "forecast_2025-06.csv", as_of="2025-05")
    tracked = track_saved_forecast(run.forecast_path, toy_df, toy_panel)
    assert tracked.run_id == run.manifest["run"]["run_id"] and tracked.month == "2025-06" and tracked.manifest_found
    assert {"sku", "planta", "total", "sin_pronostico"} <= set(tracked.scores["nivel"]) and set(tracked.rows["run_id"]) == {tracked.run_id}
    assert tracked.fired == int((tracked.alerts["tipo"] == "alerta").sum())
    csv, db = tmp_path / "seg.csv", tmp_path / "t.db"
    for _ in range(2):  # twice: replaced, not duplicated
        written = save_tracking([tracked], csv, db)
    assert written == len(tracked.rows) == len(pd.read_csv(csv)) == sqlite3.connect(db).execute("SELECT COUNT(*) FROM seguimiento").fetchone()[0]
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM corridas").fetchone()[0] == 1
    assert save_tracking([tracked], tmp_path / "only_csv.csv", None) == written  # no database


def test_a_forecast_without_manifest_is_identified_by_its_hash(data, toy_df, toy_panel, tmp_path):
    run = run_prediction(data, None, tmp_path / "forecast_2025-06.csv", as_of="2025-05")
    run.manifest_file.unlink()
    tracked = track_saved_forecast(run.forecast_path, toy_df, toy_panel)
    assert not tracked.manifest_found and tracked.run_id == f"sin-manifiesto-{file_sha256(run.forecast_path)[:8]}"


def test_a_forecast_made_with_other_data_cannot_be_tracked(data, toy_df, toy_panel, tmp_path):
    run = run_prediction(data, None, tmp_path / "forecast_2025-06.csv", as_of="2025-05")
    other = pd.read_csv(run.forecast_path).assign(planta="Z")  # none of its series exist in the data
    other.to_csv(run.forecast_path, index=False)
    with pytest.raises(SystemExit, match="does not belong to these data"):
        track_saved_forecast(run.forecast_path, toy_df, toy_panel)


def test_a_month_that_has_not_closed_cannot_be_tracked(data, toy_df, toy_panel, tmp_path):
    run = run_prediction(data, None, tmp_path / "forecast_2025-06.csv", as_of="2025-05")
    with pytest.raises(SystemExit):
        track_saved_forecast(run.forecast_path, toy_df[toy_df["fecha"] <= "2025-06-20"], build_panel(toy_df[toy_df["fecha"] <= "2025-06-20"]))


def test_the_backtest_is_tracked_from_its_cache(toy_panel, tmp_path):
    S = toy_panel
    n = S.monthly.shape[1]
    cache = collect_member_forecasts(S, make_frame(S, 1), [n - 2, n - 1], 1, {})
    path = tmp_path / "member_forecasts.csv"
    cache.to_csv(path, index=False)
    write_meta(path, {"data": panel_fingerprint(S), "n_series": len(S.keys), "horizon": 1})
    runs = track_backtest(S, path, repo_dir=tmp_path)
    assert [r.month for r in runs] == [S.labels[n - 2], S.labels[n - 1]] and {r.run_id for r in runs} == {BACKTEST_RUN}
    assert all(r.manifest["data"]["panel_fingerprint"] == panel_fingerprint(S) for r in runs)
    write_meta(path, {"data": "other", "n_series": len(S.keys), "horizon": 1})
    with pytest.raises(SystemExit):  # a cache made for other data is refused
        track_backtest(S, path)


def test_a_tuning_run_returns_the_results_and_reports_progress(toy_panel):
    steps, lines = [], []
    results = run_tuning(toy_panel, ["lgbm"], {"lgbm": 2}, n_test=3, seeds=[0], progress=lambda m, f: steps.append(f), log=lines.append)
    r = results["lgbm"]
    assert {"best_params", "base_params", "val_base", "val_best", "n_trials", "importance", "meta", "h1", "h3"} <= set(r)
    assert r["n_trials"] == 2 and r["val_best"] <= r["val_base"] and r["meta"]["tune_end"] == toy_panel.labels[toy_panel.monthly.shape[1] - 4]
    assert set(r["h1"]) >= {"base", "tuned", "naive_sku_acc"} and np.isfinite(r["h1"]["tuned"]["sku_acc"]).all()
    assert steps == sorted(steps) and steps[-1] == 1.0 and any("validation WAPE" in line for line in lines)


def test_a_tuning_run_can_leave_the_h3_check_out(toy_panel):
    results = run_tuning(toy_panel, ["lgbm"], {"lgbm": 2}, n_test=3, seeds=[0], log=lambda m: None, horizons=(1,))
    assert "h1" in results["lgbm"] and "h3" not in results["lgbm"]
