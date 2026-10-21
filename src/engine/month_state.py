"""
Is the last month of the data finished, and which forecast strategy applies.

The planner asks for the forecast at mid-month, never on a fixed day, so the cutoff is read from the data: it is the calendar day of the
last record. A month is finished when every day after the last record up to the end of the month is a Sunday (the rule of
`incomplete_last_month_message`). `resolve_strategy` turns the state and the user's settings into the strategy that runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.engine.data_checks import working_days_missing

STRATEGIES = ["automatica", "h1", "directa", "parcial", "nowcast_media", "iterada"]
NEEDS_MONTH_IN_PROGRESS = ["parcial", "nowcast_media"]
MIN_CUTOFF_DAY = 10  # from this day on the first days of the month in progress improve the direct forecast (results/curva_corte_parcial.csv)


@dataclass(frozen=True)
class MonthState:
    status: str  # "complete" or "in_progress"
    last_date: pd.Timestamp
    month: str  # month of the last record, YYYY-MM
    last_complete: str  # last finished month, YYYY-MM
    cutoff_day: int | None  # calendar day of the last record when the month is in progress
    days_seen: int  # non-Sunday days up to the last record in that month
    days_left: int  # non-Sunday days after it

    @property
    def in_progress(self) -> bool:
        return self.status == "in_progress"


@dataclass(frozen=True)
class StrategyPlan:
    strategy: str  # one of STRATEGIES except "automatica"
    cutoff_day: int | None
    origin: str  # last complete month the forecast starts from, YYYY-MM
    target: str  # month that is forecast, YYYY-MM
    notes: list[str] = field(default_factory=list)


def detect_month_state(df: pd.DataFrame) -> MonthState:
    r"""State of the last month of the table, from the date of its last record."""
    last = pd.Timestamp(df["fecha"].max())
    month = last.to_period("M")
    missing = working_days_missing(df)
    month_days = pd.date_range(month.start_time, month.end_time.normalize())
    seen = int((month_days[month_days <= last].dayofweek != 6).sum())
    if not len(missing):
        return MonthState("complete", last, str(month), str(month), None, seen, 0)
    return MonthState("in_progress", last, str(month), str(month - 1), int(last.day), seen, len(missing))


def resolve_strategy(state: MonthState, requested: str = "automatica", cutoff_day: int | None = None) -> StrategyPlan:
    r"""
    The strategy that runs for this state of the data and these settings, with the notes to show the user.

    Month complete: the one-month forecast of the month after it. Month in progress: `automatica` is `parcial` from MIN_CUTOFF_DAY and
    `directa` before it; the target is the month after the one in progress. A `cutoff_day` chosen by hand cannot pass the last record.
    """
    if requested not in STRATEGIES:
        raise ValueError(f"unknown strategy {requested!r}; choose one of {STRATEGIES}")
    origin = pd.Period(state.last_complete, freq="M")
    notes: list[str] = []
    if not state.in_progress:
        if requested not in ("automatica", "h1"):
            notes.append("El último mes está completo: se pronostica el mes siguiente con un mes de anticipación.")
        return StrategyPlan("h1", None, str(origin), str(origin + 1), notes)

    cutoff = state.cutoff_day if cutoff_day is None else int(cutoff_day)
    if cutoff < 1 or cutoff > state.cutoff_day:
        raise ValueError(f"the cutoff day must be between 1 and {state.cutoff_day} (the last record is on {state.last_date.date()})")
    strategy = requested
    if requested == "automatica":
        strategy = "parcial" if cutoff >= MIN_CUTOFF_DAY else "directa"
        if strategy == "directa":
            notes.append(f"Hay solo {cutoff} días del mes en curso: se usa la directa, que no los necesita (el aporte empieza el día {MIN_CUTOFF_DAY}).")
    elif requested in NEEDS_MONTH_IN_PROGRESS and cutoff < MIN_CUTOFF_DAY:
        notes.append(f"El corte (día {cutoff}) está bajo el rango validado, que empieza el día {MIN_CUTOFF_DAY}: el aporte del mes en curso es pequeño.")
    uses_cutoff = strategy in NEEDS_MONTH_IN_PROGRESS
    target = origin + 1 if strategy == "h1" else origin + 2
    return StrategyPlan(strategy, cutoff if uses_cutoff else None, str(origin), str(target), notes)
