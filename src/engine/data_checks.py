"""
Input checks for the consumption table (planta, sku, fecha, consumo in TO).

Three kinds of output, none of them silent:

    errors    the load stops (`InputValidationError`): missing columns or values, wrong types, negatives, repeated
              (planta, sku, fecha), keys that are not normalised or that differ only in case or spaces, a month or three
              working days in a row without any row, and (when asked) an incomplete last month.
    warnings  the load goes on and the rows are printed: a day above `PEAK_LIMIT` TO for one SKU and planta, a value far
              above the median of its series, a month with far fewer rows than the previous one, a planta that appears or
              disappears in the last month, a SKU outside the usual format.
    tables    always available with `with_tables=True`: the Sunday rows (they count as real consumption), the peak rows
              and the month-to-month change by planta and SKU and by planta (long format, one row per month).

The checks never correct anything: cleaning belongs to scripts/transform_consumos.py.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd

REQUIRED = ["planta", "sku", "fecha", "consumo"]
PEAK_LIMIT = 100.0  # TO in one day for one SKU and planta (the historical maximum is 141.7)
SPIKE_FACTOR = 20.0  # times the median of the positive days of the series
ROWS_DROP = 0.4  # a month with 40% fewer rows than the previous one
MAX_DAY_GAP = 3  # working days in a row without any row
EXAMPLES = 5  # rows printed per finding

SKU_PATTERN = re.compile(
    r"^(?P<grade>[^/]+)/(?P<subgrade>[^/]+)/(?P<gsm>\d+)gsm/(?P<ancho>\d+)mm(?:/(?P<core>[^/]+))?(?:/(?P<sufijo>.+))?$", re.IGNORECASE
)


class InputValidationError(SystemExit):
    r"""Raised when the input has errors; being a SystemExit, a script stops with the message and no traceback."""


@dataclass
class Finding:
    severity: str  # 'error' | 'warning'
    code: str
    message: str
    rows: pd.DataFrame | None = None  # the rows (or keys) that broke the rule


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self, severity: str | None = None) -> str:
        r"""One block per finding: the message and up to `EXAMPLES` rows."""
        blocks = []
        for f in self.findings:
            if severity not in (None, f.severity):
                continue
            lines = [f"{f.severity.upper()} {f.code}: {f.message}"]
            if f.rows is not None and len(f.rows):
                shown = f.rows.head(EXAMPLES).copy()
                for column in shown.columns:
                    if pd.api.types.is_datetime64_any_dtype(shown[column]):
                        shown[column] = shown[column].dt.strftime("%Y-%m-%d")
                lines += ["    " + line for line in shown.to_string(index=False).splitlines()]
                if len(f.rows) > EXAMPLES:
                    lines.append(f"    ... and {len(f.rows) - EXAMPLES} more")
            blocks.append("\n".join(lines))
        return "\n".join(blocks)


def _finding(severity: str, code: str, message: str, rows: pd.DataFrame | None = None) -> Finding:
    return Finding(severity, code, message, rows)


# ---- structure: if it fails, the other rules cannot run --------------------------------------------------------------------


def _structure(df: pd.DataFrame) -> list[Finding]:
    absent = [c for c in REQUIRED if c not in df.columns]
    if absent:
        return [_finding("error", "columns", f"missing columns {absent}; expected {REQUIRED}")]
    bad = []
    if not pd.api.types.is_datetime64_any_dtype(df["fecha"]):
        bad.append("fecha is not a date")
    if not pd.api.types.is_numeric_dtype(df["consumo"]) or pd.api.types.is_bool_dtype(df["consumo"]):
        bad.append("consumo is not a number")
    for column in ("planta", "sku"):
        if not (pd.api.types.is_string_dtype(df[column]) or pd.api.types.is_object_dtype(df[column])):
            bad.append(f"{column} is not text")
    return [_finding("error", "types", "; ".join(bad))] if bad else []


def _empty_text(series: pd.Series) -> pd.Series:
    return (series.isna() | series.astype("string").str.strip().eq("").fillna(False)).astype(bool)


def _missing(df: pd.DataFrame) -> tuple[Finding | None, pd.Series]:
    r"""The finding for empty planta, sku, fecha or consumo, and the mask of those rows."""
    mask = _empty_text(df["planta"]) | _empty_text(df["sku"]) | df["fecha"].isna() | df["consumo"].isna()
    if not mask.any():
        return None, mask
    return _finding("error", "missing", f"{int(mask.sum()):,} rows with an empty planta, sku, fecha or consumo", df.loc[mask, REQUIRED]), mask


# ---- errors ----------------------------------------------------------------------------------------------------------------


def _negative(df: pd.DataFrame, cfg) -> Finding | None:
    rows = df[df["consumo"] < 0]
    return _finding("error", "negative", f"{len(rows):,} rows with negative consumption", rows[REQUIRED]) if len(rows) else None


def _duplicate(df: pd.DataFrame, cfg) -> Finding | None:
    rows = df[df.duplicated(["planta", "sku", "fecha"], keep=False)]
    if not len(rows):
        return None
    keys = int(df.duplicated(["planta", "sku", "fecha"]).sum())
    return _finding("error", "duplicate", f"{keys:,} repeated (planta, sku, fecha) keys; they would be added up silently", rows[REQUIRED].sort_values(["planta", "sku", "fecha"]))


def _keys(df: pd.DataFrame, cfg) -> Finding | None:
    bad_planta = df["planta"] != df["planta"].str.strip().str.upper()
    bad_sku = df["sku"] != df["sku"].str.strip()
    rows = df[bad_planta | bad_sku]
    if not len(rows):
        return None
    return _finding("error", "keys", f"{len(rows):,} rows whose planta is not upper case without spaces or whose sku has spaces at the ends", rows[REQUIRED])


def _near_duplicate_keys(df: pd.DataFrame, cfg) -> Finding | None:
    pairs = df[["planta", "sku"]].drop_duplicates()
    norm = pd.DataFrame({"p": pairs["planta"].str.strip().str.upper(), "s": pairs["sku"].str.lower().str.replace(r"\s+", "", regex=True)})
    rows = pairs[norm.duplicated(keep=False)]
    return _finding("error", "near_duplicate_keys", f"{len(rows):,} series that differ only in case or spaces (they would be split in two)", rows.sort_values(["planta", "sku"])) if len(rows) else None


def _month_gap(df: pd.DataFrame, cfg) -> Finding | None:
    present = df["fecha"].dt.to_period("M").unique()
    full = pd.period_range(df["fecha"].min(), df["fecha"].max(), freq="M")
    absent = full.difference(pd.PeriodIndex(present))
    if not len(absent):
        return None
    months = [str(m) for m in absent]
    return _finding("error", "month_gap", f"months without any row between the first and the last date: {months}", pd.DataFrame({"mes": months}))


def _day_gap(df: pd.DataFrame, cfg) -> Finding | None:
    days = pd.date_range(df["fecha"].min().normalize(), df["fecha"].max().normalize())
    days = days[days.dayofweek != 6]
    absent = pd.Series(~days.isin(df["fecha"].dt.normalize().unique()))
    group = (absent != absent.shift()).cumsum()
    runs = []
    for _, part in absent.groupby(group):
        if part.iloc[0] and len(part) >= cfg.max_day_gap:
            runs.append({"desde": days[part.index[0]], "hasta": days[part.index[-1]], "dias_habiles": len(part)})
    if not runs:
        return None
    return _finding("error", "day_gap", f"{len(runs)} stretches of {cfg.max_day_gap} or more working days in a row without any row", pd.DataFrame(runs))


def incomplete_last_month_message(df: pd.DataFrame) -> str | None:
    r"""Message when the last month of the table is not finished (a day after the last record, up to the end of the month, is not a Sunday)."""
    last = df["fecha"].max()
    rest = pd.date_range(last + pd.Timedelta(days=1), last + pd.offsets.MonthEnd(0))
    missing = rest[rest.dayofweek != 6]
    if not len(missing):
        return None
    return (
        f"The last month ({last.strftime('%Y-%m')}) is incomplete: the data ends on {last.date()} and "
        f"{len(missing)} working days are missing. Export the complete month, or pass --as-of with the last complete month."
    )


def _last_month(df: pd.DataFrame, cfg) -> Finding | None:
    message = incomplete_last_month_message(df) if cfg.require_complete_last_month else None
    return _finding("error", "last_month", message) if message else None


# ---- warnings --------------------------------------------------------------------------------------------------------------


def _daily_peak(df: pd.DataFrame, cfg) -> Finding | None:
    rows = df[df["consumo"] > cfg.peak_limit].sort_values("consumo", ascending=False)
    if not len(rows):
        return None
    return _finding("warning", "daily_peak", f"{len(rows):,} rows above {cfg.peak_limit:g} TO in one day for one SKU and planta", rows[REQUIRED])


def _spike(df: pd.DataFrame, cfg) -> Finding | None:
    positive = df[df["consumo"] > 0]
    median = positive.groupby(["planta", "sku"])["consumo"].transform("median")
    ratio = positive["consumo"] / median
    mask = ratio > cfg.spike_factor
    if not mask.any():
        return None
    rows = positive[mask].assign(veces_mediana=ratio[mask].round(1)).sort_values("veces_mediana", ascending=False)
    return _finding("warning", "spike", f"{len(rows):,} rows more than {cfg.spike_factor:g} times the median of their series", rows[REQUIRED + ["veces_mediana"]])


def _rows_drop(df: pd.DataFrame, cfg) -> Finding | None:
    counts = df.groupby(df["fecha"].dt.to_period("M")).size()
    counts = counts.reindex(pd.period_range(counts.index.min(), counts.index.max(), freq="M"), fill_value=0)
    previous = counts.shift()
    mask = (previous > 0) & (counts < previous * (1 - cfg.rows_drop)) & (counts > 0)
    if not mask.any():
        return None
    rows = pd.DataFrame({"mes": counts.index[mask].astype(str), "filas": counts[mask].to_numpy(), "filas_mes_anterior": previous[mask].astype(int).to_numpy()})
    return _finding("warning", "rows_drop", f"{len(rows)} months with {cfg.rows_drop:.0%} or more fewer rows than the previous month (a partial export?)", rows)


def _planta_set(df: pd.DataFrame, cfg) -> Finding | None:
    month = df["fecha"].dt.to_period("M")
    last = month.max()
    current, earlier = set(df.loc[month == last, "planta"]), set(df.loc[month < last, "planta"])
    rows = [{"planta": p, "estado": "desaparece"} for p in sorted(earlier - current)] + [{"planta": p, "estado": "nueva"} for p in sorted(current - earlier)]
    if not rows:
        return None
    return _finding("warning", "planta_set", f"plantas that disappear or appear in the last month ({last})", pd.DataFrame(rows))


def _sku_format(df: pd.DataFrame, cfg) -> Finding | None:
    skus = pd.Series(df["sku"].unique())
    odd = skus[~skus.map(lambda s: bool(SKU_PATTERN.match(s)))]
    return _finding("warning", "sku_format", f"{len(odd)} SKUs outside the usual grade/subgrade/gsm/width format", pd.DataFrame({"sku": sorted(odd)})) if len(odd) else None


ERROR_RULES = {
    "negative": _negative,
    "duplicate": _duplicate,
    "keys": _keys,
    "near_duplicate_keys": _near_duplicate_keys,
    "month_gap": _month_gap,
    "day_gap": _day_gap,
    "last_month": _last_month,
}
WARNING_RULES = {
    "daily_peak": _daily_peak,
    "spike": _spike,
    "rows_drop": _rows_drop,
    "planta_set": _planta_set,
    "sku_format": _sku_format,
}


# ---- tables ----------------------------------------------------------------------------------------------------------------


def _by_month(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    r"""Consumption per calendar month and `keys` (rows = keys, columns = every month between the first and the last, 0 when absent)."""
    month = df["fecha"].dt.to_period("M")
    full = pd.period_range(month.min(), month.max(), freq="M")
    table = df.groupby(keys + [month.rename("mes")])["consumo"].sum().unstack("mes", fill_value=0.0)
    table = table.reindex(columns=full, fill_value=0.0)
    table.columns.name = "mes"
    return table


def _long_change(table: pd.DataFrame) -> pd.DataFrame:
    r"""From the wide table (keys x months) to one row per key and month with the previous month's value; the first month is left out."""
    columns = ["mes", *table.index.names, "consumo_mes", "consumo_mes_anterior", "cambio_TO", "cambio_pct"]
    if table.shape[1] < 2:
        return pd.DataFrame(columns=columns)
    current = table.iloc[:, 1:].copy()
    previous = table.iloc[:, :-1].copy()
    previous.columns = current.columns
    keys = list(table.index.names)
    out = current.stack().rename("consumo_mes").reset_index().merge(previous.stack().rename("consumo_mes_anterior").reset_index(), on=keys + ["mes"])
    out["mes"] = out["mes"].astype(str)
    out["cambio_TO"] = out["consumo_mes"] - out["consumo_mes_anterior"]
    out["cambio_pct"] = np.where(out["consumo_mes_anterior"] > 0, out["cambio_TO"] / out["consumo_mes_anterior"].where(out["consumo_mes_anterior"] > 0) * 100.0, np.nan)
    return out


def change_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    r"""Month-to-month change by planta and SKU (only rows where either month has consumption) and by planta, both long."""
    sku = _long_change(_by_month(df, ["planta", "sku"]))
    sku = sku[(sku["consumo_mes"] > 0) | (sku["consumo_mes_anterior"] > 0)].copy()
    sku["estado"] = np.select([sku["consumo_mes_anterior"] == 0, sku["consumo_mes"] == 0], ["nuevo", "sin consumo"], "cambia")
    sku = sku[["mes", "planta", "sku", "consumo_mes_anterior", "consumo_mes", "cambio_TO", "cambio_pct", "estado"]].sort_values(["mes", "planta", "sku"]).reset_index(drop=True)

    planta = _long_change(_by_month(df, ["planta"]))
    active = sku[sku["consumo_mes"] > 0].groupby(["mes", "planta"]).size().rename("skus_con_consumo")
    planta = planta.merge(active.reset_index(), on=["mes", "planta"], how="left").fillna({"skus_con_consumo": 0})
    planta["skus_con_consumo"] = planta["skus_con_consumo"].astype(int)
    planta = planta[["mes", "planta", "consumo_mes_anterior", "consumo_mes", "cambio_TO", "cambio_pct", "skus_con_consumo"]].sort_values(["mes", "planta"]).reset_index(drop=True)
    return {"cambio_planta_sku": sku, "cambio_planta": planta}


# ---- entry points ----------------------------------------------------------------------------------------------------------


def check_consumption(
    df: pd.DataFrame,
    *,
    peak_limit: float = PEAK_LIMIT,
    spike_factor: float = SPIKE_FACTOR,
    rows_drop: float = ROWS_DROP,
    max_day_gap: int = MAX_DAY_GAP,
    require_complete_last_month: bool = False,
    with_tables: bool = False,
) -> Report:
    r"""
    Run every rule on the consumption table and return the report; nothing is printed or raised.

    The structure rules (columns, types, empty values) run first; if the columns or types are wrong the other rules cannot run.
    With `with_tables=True` the report also holds the Sunday rows, the peak rows and the two month-to-month change tables.
    """
    report = Report()
    fatal = _structure(df)
    if fatal:
        report.findings = fatal
        return report
    missing, mask = _missing(df)
    if missing is not None:
        report.findings.append(missing)
        df = df[~mask]
    if df.empty:
        return report
    cfg = SimpleNamespace(peak_limit=peak_limit, spike_factor=spike_factor, rows_drop=rows_drop, max_day_gap=max_day_gap, require_complete_last_month=require_complete_last_month)
    for rules in (ERROR_RULES, WARNING_RULES):
        for rule in rules.values():
            found = rule(df, cfg)
            if found is not None:
                report.findings.append(found)
    if with_tables:
        report.tables["domingos"] = df[df["fecha"].dt.dayofweek == 6][REQUIRED].sort_values(["fecha", "planta", "sku"]).reset_index(drop=True)
        report.tables["picos"] = df[df["consumo"] > peak_limit][REQUIRED].sort_values("consumo", ascending=False).reset_index(drop=True)
        report.tables.update(change_tables(df))
    return report


def validate_consumption(df: pd.DataFrame, **kwargs) -> Report:
    r"""Stop with `InputValidationError` when there are errors; print the warnings (to stderr) and return the report otherwise."""
    report = check_consumption(df, **kwargs)
    if report.errors:
        raise InputValidationError("Input checks failed:\n" + report.summary("error"))
    if report.warnings:
        print("Input checks passed with warnings (details: python scripts/validate_input.py <file>):\n" + report.summary("warning"), file=sys.stderr)
    return report
