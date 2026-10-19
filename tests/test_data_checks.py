"""Input checks of the consumption table: each rule stops (error) or warns, the tables are right, and the loader validates by default."""

import numpy as np
import pandas as pd
import pytest

from src.engine.data_checks import InputValidationError, check_consumption, validate_consumption
from src.engine.panel_utils import build_panel, load_consumption

SKU1, SKU2 = "X1/01/100gsm/2000mm/1200-1400", "X2/01/120gsm/2100mm/1200-1450"


def clean(start="2024-01-01", end="2024-06-30", plantas=("AAAA", "BBBB", "CCCC")) -> pd.DataFrame:
    r"""Every series has a row on every day but Sunday: no rule fires."""
    days = pd.date_range(start, end)
    days = days[days.dayofweek != 6]
    parts = [
        pd.DataFrame({"planta": p, "sku": s, "fecha": days, "consumo": (np.arange(len(days)) % 5 + 1.0)})
        for p in plantas
        for s in (SKU1, SKU2)
    ]
    return pd.concat(parts, ignore_index=True)


def codes(report, severity=None):
    return [f.code for f in report.findings if severity in (None, f.severity)]


def test_a_clean_table_has_no_findings():
    report = check_consumption(clean())
    assert report.findings == [] and report.ok


# ---- errors: the load stops --------------------------------------------------------------------------------------------


def test_missing_columns_and_wrong_types_stop():
    assert codes(check_consumption(clean().drop(columns=["sku"]))) == ["columns"]
    assert "types" in codes(check_consumption(clean().assign(fecha=lambda d: d["fecha"].dt.strftime("%Y-%m-%d"))), "error")
    assert "types" in codes(check_consumption(clean().assign(consumo=lambda d: d["consumo"].astype(str))), "error")


def test_missing_values_and_empty_names_stop():
    df = clean()
    df.loc[3, "consumo"] = np.nan
    df.loc[5, "sku"] = "  "
    report = check_consumption(df)
    found = [f for f in report.findings if f.code == "missing"][0]
    assert found.severity == "error" and len(found.rows) == 2


def test_negative_consumption_stops():
    df = clean()
    df.loc[10, "consumo"] = -1.0
    assert codes(check_consumption(df), "error") == ["negative"]


def test_duplicate_rows_stop_and_both_copies_are_listed():
    df = pd.concat([clean(), clean().head(3)], ignore_index=True)
    found = [f for f in check_consumption(df).findings if f.code == "duplicate"][0]
    assert found.severity == "error" and len(found.rows) == 6  # 3 keys, both copies


def test_unnormalised_keys_stop():
    df = clean()
    df.loc[df["planta"] == "AAAA", "planta"] = "aaaa"
    assert "keys" in codes(check_consumption(df), "error")
    df = clean()
    df.loc[df["sku"] == SKU1, "sku"] = SKU1 + " "
    assert "keys" in codes(check_consumption(df), "error")


def test_series_that_differ_only_in_case_or_spaces_stop():
    df = clean()
    other = df[(df["planta"] == "AAAA") & (df["sku"] == SKU1)].copy()
    other["sku"] = SKU1.upper()
    other["fecha"] = other["fecha"] + pd.Timedelta(days=0)
    other = other.iloc[:5]
    assert "near_duplicate_keys" in codes(check_consumption(pd.concat([df, other], ignore_index=True)), "error")


def test_a_missing_month_stops():
    df = clean()
    df = df[df["fecha"].dt.month != 3]
    found = [f for f in check_consumption(df).findings if f.code == "month_gap"][0]
    assert found.severity == "error" and "2024-03" in found.message


def test_three_working_days_without_rows_stop_but_two_do_not():
    df = clean()
    three = df[~df["fecha"].isin(pd.to_datetime(["2024-02-05", "2024-02-06", "2024-02-07"]))]
    two = df[~df["fecha"].isin(pd.to_datetime(["2024-02-05", "2024-02-06"]))]
    assert "day_gap" in codes(check_consumption(three), "error")
    assert "day_gap" not in codes(check_consumption(two))


def test_an_incomplete_last_month_stops_only_when_asked():
    partial = clean()
    partial = partial[partial["fecha"] <= "2024-06-20"]
    assert "last_month" not in codes(check_consumption(partial))
    assert "last_month" in codes(check_consumption(partial, require_complete_last_month=True), "error")
    assert "last_month" not in codes(check_consumption(clean(), require_complete_last_month=True))


# ---- warnings: the load goes on --------------------------------------------------------------------------------------------


def test_daily_peak_warns_with_planta_sku_and_day_and_does_not_stop(capsys):
    df = clean()
    df.loc[(df["planta"] == "BBBB") & (df["sku"] == SKU2) & (df["fecha"] == "2024-04-09"), "consumo"] = 141.7
    report = check_consumption(df, with_tables=True)
    peak = [f for f in report.findings if f.code == "daily_peak"][0]
    assert peak.severity == "warning" and report.ok
    assert peak.rows[["planta", "sku", "fecha", "consumo"]].iloc[0].tolist() == ["BBBB", SKU2, pd.Timestamp("2024-04-09"), 141.7]
    assert len(report.tables["picos"]) == 1
    validate_consumption(df)  # prints, does not raise
    err = capsys.readouterr().err
    assert "daily_peak" in err and "BBBB" in err and SKU2 in err and "2024-04-09" in err


def test_a_value_equal_to_the_limit_is_not_a_peak():
    df = clean()
    df.loc[0, "consumo"] = 100.0
    assert "daily_peak" not in codes(check_consumption(df))
    assert "daily_peak" in codes(check_consumption(df, peak_limit=99.0))


def test_spike_against_the_median_of_the_series():
    df = clean()
    df.loc[7, "consumo"] = 70.0  # about 23 times the median of its series, below the peak limit
    assert codes(check_consumption(df), "warning") == ["spike"]


def test_a_month_with_far_fewer_rows_warns():
    df = clean()
    april = df["fecha"].dt.month == 4
    keep = ~april | ((df["planta"] == "AAAA") & (df["sku"] == SKU1))
    found = [f for f in check_consumption(df[keep]).findings if f.code == "rows_drop"]
    assert found and "2024-04" in found[0].rows["mes"].tolist()


def test_a_planta_that_disappears_or_appears_in_the_last_month_warns():
    df = clean()
    df = df[~((df["planta"] == "CCCC") & (df["fecha"] >= "2024-06-01"))]
    new = clean(start="2024-06-03", end="2024-06-29", plantas=("DDDD",))
    report = check_consumption(pd.concat([df, new], ignore_index=True))
    found = [f for f in report.findings if f.code == "planta_set"][0]
    assert dict(zip(found.rows["planta"], found.rows["estado"])) == {"CCCC": "desaparece", "DDDD": "nueva"}


def test_a_sku_outside_the_expected_format_warns():
    df = clean()
    df.loc[df["sku"] == SKU1, "sku"] = "SF1/200GSM/2100MM"
    found = [f for f in check_consumption(df).findings if f.code == "sku_format"][0]
    assert found.severity == "warning" and found.rows["sku"].tolist() == ["SF1/200GSM/2100MM"]


# ---- tables ---------------------------------------------------------------------------------------------------------------


def test_sunday_rows_table_has_every_sunday_row_and_counts_as_consumption():
    df = pd.concat([clean(), pd.DataFrame({"planta": ["AAAA", "BBBB"], "sku": [SKU1, SKU2], "fecha": pd.to_datetime(["2024-03-03", "2024-03-10"]), "consumo": [1.5, 2.5]})], ignore_index=True)
    report = check_consumption(df, with_tables=True)
    sundays = report.tables["domingos"]
    assert list(sundays.columns) == ["planta", "sku", "fecha", "consumo"] and len(sundays) == 2 and sundays["consumo"].sum() == 4.0
    assert report.ok and codes(report) == []  # no warning for Sundays
    assert build_panel(df).monthly.shape[1] == 6  # still part of the monthly totals


def test_change_tables_states_and_values():
    rows = [
        ("A", "s1", "2024-01-10", 10.0), ("A", "s1", "2024-02-10", 15.0),  # cambia
        ("A", "s2", "2024-02-12", 5.0),  # nuevo
        ("A", "s3", "2024-01-11", 4.0),  # sin consumo
        ("B", "s4", "2024-01-15", 8.0), ("B", "s4", "2024-02-15", 8.0),
    ]
    df = pd.DataFrame(rows, columns=["planta", "sku", "fecha", "consumo"]).assign(fecha=lambda d: pd.to_datetime(d["fecha"]))
    t = check_consumption(df, with_tables=True).tables
    sku = t["cambio_planta_sku"].set_index(["planta", "sku"])
    assert set(t["cambio_planta_sku"]["mes"]) == {"2024-02"}  # the first month has no previous one
    assert sku.loc[("A", "s1"), ["consumo_mes_anterior", "consumo_mes", "cambio_TO", "cambio_pct", "estado"]].tolist() == [10.0, 15.0, 5.0, 50.0, "cambia"]
    assert sku.loc[("A", "s2"), "estado"] == "nuevo" and np.isnan(sku.loc[("A", "s2"), "cambio_pct"])
    assert sku.loc[("A", "s3"), ["cambio_TO", "cambio_pct", "estado"]].tolist() == [-4.0, -100.0, "sin consumo"]
    planta = t["cambio_planta"].set_index("planta")
    assert planta.loc["A", ["consumo_mes_anterior", "consumo_mes", "cambio_TO", "skus_con_consumo"]].tolist() == [14.0, 20.0, 6.0, 2]


def test_planta_change_is_the_sum_of_its_sku_changes():
    rng = np.random.default_rng(0)
    df = clean()
    df["consumo"] = rng.integers(0, 6, len(df)) * 1.0
    df = df[df["consumo"] > 0]
    t = check_consumption(df, with_tables=True).tables
    by_sku = t["cambio_planta_sku"].groupby(["mes", "planta"])["cambio_TO"].sum()
    by_planta = t["cambio_planta"].set_index(["mes", "planta"])["cambio_TO"]
    assert np.allclose(by_sku.reindex(by_planta.index).fillna(0.0), by_planta)
    assert len(t["cambio_planta"]) == 3 * 5  # 3 plantas, 5 months with a previous one


def test_change_tables_use_calendar_months_so_a_gap_is_not_skipped():
    df = clean()
    df = df[df["fecha"].dt.month != 3]
    t = check_consumption(df, with_tables=True).tables["cambio_planta"]
    march = t[t["mes"] == "2024-03"]
    assert (march["consumo_mes"] == 0).all() and len(march) == 3


# ---- the load -------------------------------------------------------------------------------------------------------------


def test_validate_raises_system_exit_with_the_codes_and_examples():
    df = pd.concat([clean(), clean().head(2)], ignore_index=True)
    with pytest.raises(InputValidationError) as info:
        validate_consumption(df)
    assert isinstance(info.value, SystemExit) and "duplicate" in str(info.value.code)


def test_load_consumption_validates_by_default_and_can_skip(tmp_path):
    good, bad = tmp_path / "good.csv", tmp_path / "bad.csv"
    clean().to_csv(good, index=False)
    pd.concat([clean(), clean().head(5)]).to_csv(bad, index=False)
    with pytest.raises(InputValidationError):
        load_consumption(bad)
    assert len(load_consumption(bad, validate=False)) == len(clean()) + 5
    assert len(load_consumption(good)) == len(clean())
