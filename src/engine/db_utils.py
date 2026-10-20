"""
SQLite store for the real consumption and the predicted consumption.

Tables: `plantas`, `skus`, `consumos` (real, daily), `consumos_predichos` (monthly forecasts), `corridas` (one row per forecast run, from its
manifest) and `seguimiento` (accuracy of a closed month against the real, per run). The csv stays the master of the real data; the database is
built from it (scripts/build_database.py). While the project is in beta the predictions are overwritten: a load replaces the whole block (same
`estrategia` and `mes_origen`) and stamps `creado_en`. The tracking history is also kept in results/seguimiento.csv, which goes to git, so a
rebuilt database gets it back (`load_tracking_csv`).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.engine.data_checks import SKU_PATTERN, check_consumption
from src.engine.panel_utils import Panel

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS plantas (
    planta_id  INTEGER PRIMARY KEY,
    codigo     TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS skus (
    sku_id              INTEGER PRIMARY KEY,
    descripcion         TEXT NOT NULL UNIQUE,
    grade               TEXT,
    subgrade            TEXT,
    gsm                 INTEGER,
    ancho_mm            INTEGER,
    ancho_core          TEXT,
    sufijo              TEXT,
    clave_planificador  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS consumos (
    planta_id  INTEGER NOT NULL REFERENCES plantas (planta_id),
    sku_id     INTEGER NOT NULL REFERENCES skus (sku_id),
    fecha      TEXT    NOT NULL CHECK (fecha GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    consumo    REAL    NOT NULL CHECK (consumo >= 0),
    PRIMARY KEY (planta_id, sku_id, fecha)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_consumos_fecha ON consumos (fecha);

CREATE TABLE IF NOT EXISTS consumos_predichos (
    planta_id     INTEGER NOT NULL REFERENCES plantas (planta_id),
    sku_id        INTEGER NOT NULL REFERENCES skus (sku_id),
    mes_objetivo  TEXT    NOT NULL CHECK (mes_objetivo GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]'),
    mes_origen    TEXT    NOT NULL CHECK (mes_origen GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]' AND mes_origen < mes_objetivo),
    estrategia    TEXT    NOT NULL,
    modelo        TEXT    NOT NULL,
    valor         REAL    NOT NULL CHECK (valor >= 0),
    creado_en     TEXT    NOT NULL,
    run_id        TEXT,
    PRIMARY KEY (planta_id, sku_id, mes_objetivo, mes_origen, estrategia, modelo)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_predichos_objetivo ON consumos_predichos (mes_objetivo);

CREATE TABLE IF NOT EXISTS corridas (
    run_id            TEXT PRIMARY KEY,
    creado_en         TEXT NOT NULL,
    mes_pronosticado  TEXT,
    as_of             TEXT,
    commit_git        TEXT,
    huella_datos      TEXT,
    params_hash       TEXT,
    csv_sha256        TEXT,
    total_ml_mean     REAL
);

CREATE TABLE IF NOT EXISTS seguimiento (
    run_id          TEXT NOT NULL,
    mes             TEXT NOT NULL CHECK (mes GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]'),
    nivel           TEXT NOT NULL,
    planta          TEXT NOT NULL DEFAULT '',
    modelo          TEXT NOT NULL,
    acierto         REAL,
    wape            REAL,
    bias_pct        REAL,
    n               INTEGER,
    real_TO         REAL,
    pronostico_TO   REAL,
    alerta          TEXT NOT NULL DEFAULT '',
    creado_en       TEXT NOT NULL,
    PRIMARY KEY (run_id, mes, nivel, planta, modelo)
) WITHOUT ROWID;

CREATE VIEW IF NOT EXISTS consumo_mensual AS
SELECT planta_id, sku_id, substr(fecha, 1, 7) AS mes, SUM(consumo) AS consumo, COUNT(*) AS dias_con_consumo
FROM consumos
GROUP BY planta_id, sku_id, mes;
"""

PREDICTION_KEY = ["planta", "sku", "mes_objetivo", "mes_origen", "estrategia", "modelo"]


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    r"""Open the database with foreign keys enforced (SQLite leaves them off by default)."""
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA foreign_keys = ON")
    return con


def create_schema(con: sqlite3.Connection) -> None:
    r"""Create the tables, indexes and the monthly view if they do not exist, and migrate an older database to the current version."""
    con.executescript(SCHEMA)
    columns = [row[1] for row in con.execute("PRAGMA table_info(consumos_predichos)")]
    if "run_id" not in columns:  # version 1 -> 2: only adds, nothing is deleted
        con.execute("ALTER TABLE consumos_predichos ADD COLUMN run_id TEXT")
    con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    con.commit()


def parse_sku(description: str) -> dict:
    r"""
    Split a material description such as `HP2/01/115gsm/1790mm/1200-1400/Subst.` into its parts.

    Descriptions that do not follow the pattern (e.g. `SF1/200GSM/2100MM`) keep every part empty. The planner key is
    the first four parts of the description in lowercase, as in the planner comparison.
    """
    match = SKU_PATTERN.match(description)
    parts = match.groupdict() if match else {}
    return {
        "descripcion": description,
        "grade": parts.get("grade"),
        "subgrade": parts.get("subgrade"),
        "gsm": int(parts["gsm"]) if parts.get("gsm") else None,
        "ancho_mm": int(parts["ancho"]) if parts.get("ancho") else None,
        "ancho_core": parts.get("core"),
        "sufijo": parts.get("sufijo"),
        "clave_planificador": "/".join(description.split("/")[:4]).lower(),
    }


def _id_map(con: sqlite3.Connection, table: str, key: str, column: str) -> dict[str, int]:
    return {name: i for i, name in con.execute(f"SELECT {key}, {column} FROM {table}")}


def write_consumption(con: sqlite3.Connection, records: pd.DataFrame) -> dict[str, int]:
    r"""
    Replace the real consumption with `records` (planta, sku, fecha, consumo).

    New plantas and SKUs are added and never deleted, so the predictions that point to them stay valid. A repeated
    (planta, sku, fecha) is an error. Runs in one transaction. Returns the row counts of the three tables.
    """
    report = check_consumption(records)
    if report.errors:
        raise ValueError("records failed the input checks:\n" + report.summary("error"))

    dates = pd.to_datetime(records["fecha"]).dt.strftime("%Y-%m-%d")
    with con:
        con.executemany("INSERT OR IGNORE INTO plantas (codigo) VALUES (?)", [(p,) for p in sorted(records["planta"].unique())])
        sku_rows = [parse_sku(s) for s in sorted(records["sku"].unique())]
        con.executemany(
            "INSERT OR IGNORE INTO skus (descripcion, grade, subgrade, gsm, ancho_mm, ancho_core, sufijo, clave_planificador) "
            "VALUES (:descripcion, :grade, :subgrade, :gsm, :ancho_mm, :ancho_core, :sufijo, :clave_planificador)",
            sku_rows,
        )
        planta_id, sku_id = _id_map(con, "plantas", "planta_id", "codigo"), _id_map(con, "skus", "sku_id", "descripcion")
        con.execute("DELETE FROM consumos")
        con.executemany(
            "INSERT INTO consumos (planta_id, sku_id, fecha, consumo) VALUES (?, ?, ?, ?)",
            zip(records["planta"].map(planta_id), records["sku"].map(sku_id), dates, records["consumo"].astype(float)),
        )
    return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("plantas", "skus", "consumos")}


def read_consumption(con: sqlite3.Connection) -> pd.DataFrame:
    r"""The real consumption as the table `load_consumption` returns: planta, sku, fecha (datetime), consumo."""
    return pd.read_sql(
        "SELECT p.codigo AS planta, s.descripcion AS sku, c.fecha AS fecha, c.consumo AS consumo "
        "FROM consumos c JOIN plantas p USING (planta_id) JOIN skus s USING (sku_id)",
        con,
        parse_dates=["fecha"],
    )


def now_utc() -> str:
    r"""Timestamp written in `creado_en`: UTC, to the second."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def replace_predictions(con: sqlite3.Connection, preds: pd.DataFrame, created_at: str | None = None, run_id: str | None = None) -> int:
    r"""
    Overwrite the predictions of every block (`estrategia`, `mes_origen`) present in `preds`.

    `preds` has planta, sku, mes_objetivo, mes_origen, estrategia, modelo and valor (TO). The old rows of each block are
    deleted and the new ones inserted in one transaction, so a failed load keeps the previous version and a SKU that
    disappeared from the new run leaves no stale row. Every row gets `creado_en` = `created_at` (now, UTC, by default) and the `run_id` of the
    forecast run that made it, when given.
    Returns the number of rows written.
    """
    missing = set(PREDICTION_KEY + ["valor"]) - set(preds.columns)
    if missing:
        raise ValueError(f"preds lacks the columns {sorted(missing)}")
    if preds.duplicated(PREDICTION_KEY).any():
        raise ValueError("preds has repeated keys")
    if not np.isfinite(preds["valor"].to_numpy(dtype=float)).all():
        raise ValueError("preds has missing or infinite values")
    planta_id, sku_id = _id_map(con, "plantas", "planta_id", "codigo"), _id_map(con, "skus", "sku_id", "descripcion")
    unknown_p, unknown_s = set(preds["planta"]) - set(planta_id), set(preds["sku"]) - set(sku_id)
    if unknown_p or unknown_s:
        raise ValueError(f"unknown plantas {sorted(unknown_p)[:5]} or skus {sorted(unknown_s)[:5]}: load the consumption first")

    stamp = created_at or now_utc()
    blocks = preds[["estrategia", "mes_origen"]].drop_duplicates().itertuples(index=False, name=None)
    rows = zip(
        preds["planta"].map(planta_id), preds["sku"].map(sku_id), preds["mes_objetivo"], preds["mes_origen"],
        preds["estrategia"], preds["modelo"], preds["valor"].astype(float), [stamp] * len(preds), [run_id] * len(preds),
    )
    with con:
        con.executemany("DELETE FROM consumos_predichos WHERE estrategia = ? AND mes_origen = ?", list(blocks))
        con.executemany(
            "INSERT INTO consumos_predichos (planta_id, sku_id, mes_objetivo, mes_origen, estrategia, modelo, valor, creado_en, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(preds)


def read_predictions(con: sqlite3.Connection, estrategia: str | None = None, modelo: str | None = None) -> pd.DataFrame:
    r"""The predictions with planta and sku names, optionally for one `estrategia` and/or `modelo`."""
    where, args = [], []
    for column, value in (("estrategia", estrategia), ("modelo", modelo)):
        if value is not None:
            where.append(f"q.{column} = ?")
            args.append(value)
    sql = (
        "SELECT p.codigo AS planta, s.descripcion AS sku, q.mes_objetivo, q.mes_origen, q.estrategia, q.modelo, q.valor, q.creado_en, q.run_id "
        "FROM consumos_predichos q JOIN plantas p USING (planta_id) JOIN skus s USING (sku_id)"
    )
    if where:
        sql += " WHERE " + " AND ".join(where)
    return pd.read_sql(sql, con, params=args)


def predictions_from_cache(frame: pd.DataFrame, S: Panel, columns: dict[str, tuple[str, str]]) -> pd.DataFrame:
    r"""
    Long predictions from a wide cache (series, o, t and one column per forecast).

    `columns` maps a column of `frame` to (estrategia, modelo). The series index is translated with `S.keys`, so the
    caller must have checked the alignment of the cache with the panel first (cache_utils.check_alignment).
    """
    keys = np.asarray(S.keys, dtype=object)[frame["series"].to_numpy()]
    labels = np.asarray(S.labels, dtype=object)
    base = pd.DataFrame(
        {"planta": keys[:, 0], "sku": keys[:, 1], "mes_objetivo": labels[frame["t"].to_numpy()], "mes_origen": labels[frame["o"].to_numpy()]}
    )
    parts = []
    for column, (estrategia, modelo) in columns.items():
        part = base.assign(estrategia=estrategia, modelo=modelo, valor=frame[column].to_numpy(dtype=float))
        parts.append(part.dropna(subset=["valor"]))
    return pd.concat(parts, ignore_index=True)


def predictions_from_forecast(frame: pd.DataFrame, estrategia: str = "h1") -> pd.DataFrame:
    r"""Long predictions from the output of predict_next_month.py (planta, sku, mes and one column per model)."""
    models = [c for c in frame.columns if c not in ("planta", "sku", "mes")]
    origin = (pd.PeriodIndex(frame["mes"], freq="M") - 1).astype(str)
    base = pd.DataFrame({"planta": frame["planta"], "sku": frame["sku"], "mes_objetivo": frame["mes"], "mes_origen": origin})
    return pd.concat(
        [base.assign(estrategia=estrategia, modelo=m, valor=frame[m].astype(float)) for m in models], ignore_index=True
    )


TRACKING_COLUMNS = ["run_id", "mes", "nivel", "planta", "modelo", "acierto", "wape", "bias_pct", "n", "real_TO", "pronostico_TO", "alerta", "creado_en"]


def upsert_run(con: sqlite3.Connection, manifest: dict) -> None:
    r"""Store (or replace) the row of a forecast run from its manifest (src/engine/run_manifest.py); missing fields are left empty."""
    run, code, data = manifest.get("run", {}), manifest.get("code", {}), manifest.get("data", {})
    params, output = manifest.get("params", {}), manifest.get("output", {})
    with con:
        con.execute(
            "INSERT OR REPLACE INTO corridas (run_id, creado_en, mes_pronosticado, as_of, commit_git, huella_datos, params_hash, csv_sha256, total_ml_mean) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run["run_id"], run.get("created_at") or now_utc(), run.get("forecast_month"), run.get("as_of"), code.get("commit"), data.get("panel_fingerprint"),
             params.get("params_hash"), output.get("csv_sha256"), output.get("total_ml_mean")),
        )


def replace_tracking(con: sqlite3.Connection, rows: pd.DataFrame) -> int:
    r"""Replace the tracking rows of every (run_id, mes) present in `rows` (the columns of TRACKING_COLUMNS) in one transaction."""
    missing = set(TRACKING_COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"rows lacks the columns {sorted(missing)}")
    frame = rows[TRACKING_COLUMNS].astype(object).where(rows[TRACKING_COLUMNS].notna(), None)
    keys = rows[["run_id", "mes"]].drop_duplicates().itertuples(index=False, name=None)
    with con:
        con.executemany("DELETE FROM seguimiento WHERE run_id = ? AND mes = ?", list(keys))
        con.executemany(f"INSERT INTO seguimiento ({', '.join(TRACKING_COLUMNS)}) VALUES ({', '.join('?' * len(TRACKING_COLUMNS))})", frame.itertuples(index=False, name=None))
    return len(rows)


def read_tracking_csv(path: str | Path) -> pd.DataFrame:
    r"""The tracking history file (results/seguimiento.csv) with empty planta and alerta as empty strings and no infinite values (read as empty)."""
    table = pd.read_csv(path, dtype={"planta": str, "alerta": str}).fillna({"planta": "", "alerta": ""})
    return table.replace([np.inf, -np.inf], np.nan)


def load_tracking_csv(con: sqlite3.Connection, path: str | Path) -> int:
    r"""Put the history of the csv into the database (after a rebuild); rows already there for the same run and month are replaced."""
    return replace_tracking(con, read_tracking_csv(path))
