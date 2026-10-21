"""State of the last month from the date of the last record, and the strategy that runs for it."""

import pandas as pd
import pytest

from src.engine.month_state import MIN_CUTOFF_DAY, STRATEGIES, detect_month_state, resolve_strategy


def _table(last: str) -> pd.DataFrame:
    return pd.DataFrame({"planta": ["A", "A"], "sku": ["S", "S"], "fecha": pd.to_datetime(["2026-07-31", last]), "consumo": [1.0, 2.0]})


def test_month_finished_on_its_last_day():
    state = detect_month_state(_table("2026-09-30"))
    assert state.status == "complete" and not state.in_progress
    assert state.month == state.last_complete == "2026-09" and state.cutoff_day is None and state.days_left == 0


def test_month_ending_in_sundays_is_finished():
    assert detect_month_state(_table("2026-05-30")).status == "complete"  # Saturday; the 31st is a Sunday


def test_month_in_progress_reads_the_cutoff_from_the_last_record():
    state = detect_month_state(_table("2026-10-15"))
    assert state.in_progress and state.month == "2026-10" and state.last_complete == "2026-09"
    assert state.cutoff_day == 15
    assert state.days_seen == 13 and state.days_left == 14  # 15 days seen minus the Sundays of the 4th and 11th; 16 left minus the 18th and 25th


def test_a_saturday_or_a_missing_monday_leaves_the_month_in_progress():
    assert detect_month_state(_table("2026-08-29")).in_progress  # Saturday, Monday the 31st still missing
    assert detect_month_state(_table("2026-09-26")).days_left == 3  # 28, 29, 30


def test_complete_month_runs_the_one_month_forecast():
    plan = resolve_strategy(detect_month_state(_table("2026-09-30")))
    assert (plan.strategy, plan.origin, plan.target, plan.cutoff_day, plan.notes) == ("h1", "2026-09", "2026-10", None, [])


def test_other_choices_fall_back_to_one_month_when_the_month_is_complete():
    plan = resolve_strategy(detect_month_state(_table("2026-09-30")), "parcial")
    assert plan.strategy == "h1" and plan.notes


def test_automatic_uses_the_partial_strategy_from_the_validated_cutoff():
    plan = resolve_strategy(detect_month_state(_table("2026-10-15")))
    assert (plan.strategy, plan.cutoff_day, plan.origin, plan.target) == ("parcial", 15, "2026-09", "2026-11")
    assert resolve_strategy(detect_month_state(_table(f"2026-10-{MIN_CUTOFF_DAY:02d}"))).strategy == "parcial"


def test_automatic_falls_back_to_direct_under_the_validated_cutoff():
    plan = resolve_strategy(detect_month_state(_table("2026-10-08")))
    assert (plan.strategy, plan.cutoff_day, plan.target) == ("directa", None, "2026-11") and plan.notes


def test_a_chosen_strategy_under_the_range_runs_with_a_warning():
    plan = resolve_strategy(detect_month_state(_table("2026-10-08")), "nowcast_media")
    assert plan.strategy == "nowcast_media" and plan.cutoff_day == 8 and plan.notes


def test_manual_cutoff_is_used_and_cannot_pass_the_last_record():
    state = detect_month_state(_table("2026-10-22"))
    assert resolve_strategy(state, cutoff_day=12).cutoff_day == 12
    with pytest.raises(ValueError, match="between 1 and 22"):
        resolve_strategy(state, cutoff_day=23)
    with pytest.raises(ValueError):
        resolve_strategy(state, cutoff_day=0)


def test_one_month_choice_with_the_month_in_progress_forecasts_the_month_in_progress():
    plan = resolve_strategy(detect_month_state(_table("2026-10-15")), "h1")
    assert (plan.strategy, plan.origin, plan.target) == ("h1", "2026-09", "2026-10")


def test_direct_and_iterated_do_not_use_the_cutoff():
    state = detect_month_state(_table("2026-10-15"))
    assert resolve_strategy(state, "directa").cutoff_day is None and resolve_strategy(state, "iterada").cutoff_day is None


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        resolve_strategy(detect_month_state(_table("2026-10-15")), "otra")
    assert "automatica" in STRATEGIES
