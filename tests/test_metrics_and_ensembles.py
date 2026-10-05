"""Metrics, ensembles, reconciliation and the Model Confidence Set."""

import numpy as np
import pandas as pd
import pytest

from src.utils.ensemble_utils import add_ensembles, combine, inverse_error_weights
from src.utils.evaluation_utils import reconcile
from src.utils.feature_utils import direct_forecasts, make_frame
from src.utils.mcs_utils import model_confidence_set
from src.utils.metrics_utils import rank_by_wape, score

MEMBERS = ["naive", "mean3", "mean6"]


@pytest.fixture(scope="module")
def results(toy_panel):
    frame = make_frame(toy_panel, 1)
    columns = ["series", "o", "t", "planta_code", "y"] + [f"p_{m}" for m in MEMBERS]
    return frame[frame["t"] >= 22][columns].reset_index(drop=True)


def test_score_definitions_and_sign():
    y, p = np.array([10.0, 10.0]), np.array([8.0, 14.0])
    out = score(y, p)
    assert out["wape"] == pytest.approx(6 / 20)
    assert out["accuracy"] == pytest.approx(1 - 6 / 20)
    assert out["bias_pct"] == pytest.approx((20 - 22) / 20 * 100)
    assert score(np.array([10.0]), np.array([8.0]))["bias_pct"] > 0  # forecast too low: positive bias


def test_ranking_is_ascending_and_ties_share_a_rank():
    rows = [{"model": m, **score(np.array([10.0]), np.array([p]))} for m, p in [("a", 9.0), ("b", 11.0), ("c", 5.0)]]
    table = rank_by_wape(rows)
    assert list(table["rank"]) == [1, 1, 3]
    assert list(table.index)[-1] == "c"


def test_combine_methods():
    P = np.array([[1.0, 2.0, 9.0], [4.0, 4.0, 4.0]])
    assert np.allclose(combine(P, "mean"), [4.0, 4.0])
    assert np.allclose(combine(P, "median"), [2.0, 4.0])
    assert np.allclose(combine(P, "weighted", np.array([0.5, 0.5, 0.0])), [1.5, 4.0])


def test_inverse_error_weights_sum_to_one_and_favour_the_better_member():
    w = inverse_error_weights(np.array([[1.0, 4.0], [1.0, 4.0]]))
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1]


def test_trimmed_equals_mean_with_three_members(results):
    out = add_ensembles(results, {"g": MEMBERS}, 1)
    assert np.allclose(out["p_g_trimmed"], out["p_g_mean"])  # int(0.2 * 3) = 0 values trimmed


def test_first_month_has_equal_weights_and_weights_use_only_known_months(results):
    out = add_ensembles(results, {"g": MEMBERS}, 1)
    first = out["t"] == out["t"].min()
    assert np.allclose(out.loc[first, "p_g_weighted"], out.loc[first, "p_g_mean"])

    # Corrupting the actuals and forecasts of the last month must not change the earlier months.
    broken = results.copy()
    last = broken["t"] == broken["t"].max()
    broken.loc[last, [f"p_{m}" for m in MEMBERS]] *= 7
    broken.loc[last, "y"] *= 5
    other = add_ensembles(broken, {"g": MEMBERS}, 1)
    earlier = out["t"] < out["t"].max()
    assert np.allclose(out.loc[earlier, "p_g_weighted"], other.loc[earlier, "p_g_weighted"])


def test_reconcile_makes_skus_add_up_to_the_planta_control_total(toy_panel, results):
    S = toy_panel
    months = sorted(results["t"].unique())
    out = results.assign(p_rec=reconcile(results, S, 1, months, "p_mean3"))
    for t in months:
        control = direct_forecasts(S.planta_monthly, t - 1, 1, S.business_days)["mean6_x_seasonal"]
        sums = out[out["t"] == t].groupby("planta_code")["p_rec"].sum()
        for planta, total in sums.items():
            if np.isfinite(control[planta]):
                assert total == pytest.approx(control[planta])


def test_mcs_drops_a_clearly_worse_model_and_keeps_the_best():
    rng = np.random.default_rng(0)
    periods = np.repeat(np.arange(24), 6)
    month_level = np.repeat(rng.normal(0, 2, 24), 6)
    loss = pd.DataFrame(
        {
            "good": 10 + month_level + rng.normal(0, 1, len(periods)),
            "same": 10 + month_level + rng.normal(0, 1, len(periods)),
            "bad": 14 + month_level + rng.normal(0, 1, len(periods)),
        }
    )
    table = model_confidence_set(loss, periods, block=3, n_boot=1000, seed=1)
    assert table.loc["bad", "mcs_p_value"] < 0.10
    assert table["mcs_p_value"].max() == 1.0
    assert table.loc["good", "mcs_p_value"] > 0.10 and table.loc["same", "mcs_p_value"] > 0.10
    assert list(table["mcs_p_value"]) == sorted(table["mcs_p_value"])  # p-values never decrease along the elimination order
