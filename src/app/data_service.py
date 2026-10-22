"""Loading new SAP exports from the app: stage the files, review them against the current master and apply with a backup.

The review reads the exports with the same reader and the same `drop_zeros` rule that built `data/consumos_long.csv`, adds them to the current table
(a day that already exists must have the same value, or the user chooses which one is right) and runs the input checks of `data_checks` on the result.
Nothing is written to the master until the user applies a review that has no errors; the previous master is kept in `data/respaldo/`.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from src.app.paths import AppPaths
from src.engine.data_checks import MAX_DAY_GAP, PEAK_LIMIT, ROWS_DROP, SPIKE_FACTOR, check_consumption
from src.engine.fingerprints import file_sha256
from src.engine.transform_utils import read_export, write_csv

KEYS = ["planta", "sku", "fecha"]
COLUMNS = [*KEYS, "consumo"]
TOLERANCE = 1e-9
KEEP_BACKUPS = 10
STALE_HOURS = 24
POLICIES = ("detener", "usar_nuevo")
EXAMPLES = 5

TITLES = {
    "columns": "Faltan columnas", "types": "Tipos de datos incorrectos", "missing": "Filas con datos vacíos", "negative": "Consumos negativos",
    "duplicate": "Días repetidos para el mismo SKU y planta", "keys": "Nombres de planta o SKU sin normalizar", "near_duplicate_keys": "Series que solo difieren en mayúsculas o espacios",
    "month_gap": "Meses sin ninguna fila", "day_gap": "Tramos largos sin ninguna fila", "last_month": "El último mes está incompleto",
    "daily_peak": "Días con un consumo muy alto", "spike": "Valores mucho mayores que lo normal de su serie", "rows_drop": "Meses con muchas menos filas que el anterior",
    "planta_set": "Plantas que aparecen o desaparecen en el último mes", "sku_format": "SKU fuera del formato habitual",
    "conflictos": "El exporte trae valores distintos para días ya cargados", "mes_parcial": "El último mes de los datos quedaría incompleto",
    "sin_cambios": "El exporte no trae nada nuevo",
}
ADVICE = {
    "duplicate": "Revisa el exporte: el mismo día aparece más de una vez.", "month_gap": "Falta un mes completo entre los datos; el modelo necesita meses seguidos.",
    "daily_peak": "Solo es un aviso: confirma que no sea un error de SAP.", "spike": "Solo es un aviso: confirma que no sea un error de SAP.",
    "mes_parcial": "Se puede cargar. El pronóstico usará los días que lleva ese mes (desde el día 10 mejoran el resultado); los resúmenes de consumo lo dejan fuera hasta que termine.",
}


def _spanish(code: str, count: int | None, rows: pd.DataFrame | None, english: str) -> str:
    r"""The message of an input check in Spanish (the checks of the engine are written in English); unknown codes keep the original text."""
    n = f"{count:,}".replace(",", ".") if count is not None else ""
    texts = {
        "columns": "Faltan columnas en los datos.", "types": "Hay columnas con un tipo de dato incorrecto.",
        "missing": f"{n} filas con planta, SKU, día o consumo vacío.", "negative": f"{n} filas con consumo negativo.",
        "duplicate": f"{n} combinaciones de planta, SKU y día aparecen repetidas; se sumarían sin avisar.",
        "keys": f"{n} filas con la planta sin normalizar (va en mayúsculas y sin espacios) o con espacios en el SKU.",
        "near_duplicate_keys": f"{n} series solo difieren en mayúsculas o espacios; quedarían separadas en dos.",
        "month_gap": "Hay meses sin ninguna fila entre el primer y el último día" + (": " + ", ".join(map(str, rows["mes"])) if rows is not None and "mes" in rows else "") + ".",
        "day_gap": f"{n} tramos de {MAX_DAY_GAP} o más días hábiles seguidos sin ninguna fila.",
        "daily_peak": f"{n} filas con más de {PEAK_LIMIT:g} TO en un día para un SKU y planta.",
        "spike": f"{n} filas con más de {SPIKE_FACTOR:g} veces la mediana de su serie.",
        "rows_drop": f"{n} meses con {ROWS_DROP:.0%} o más filas menos que el mes anterior (¿un exporte parcial?).",
        "planta_set": "Hay plantas que aparecen o desaparecen en el último mes.", "sku_format": f"{n} SKU fuera del formato habitual grado/subgrado/gramaje/ancho.",
    }
    return texts.get(code, english)


def _partial_month(combined: pd.DataFrame) -> str | None:
    r"""Spanish message when the last month of the table is not finished (a day after the last record, up to the end of the month, is not a Sunday)."""
    last = combined["fecha"].max()
    rest = pd.date_range(last + pd.Timedelta(days=1), last + pd.offsets.MonthEnd(0))
    missing = rest[rest.dayofweek != 6]
    if not len(missing):
        return None
    return f"Los datos de {last.strftime('%Y-%m')} terminan el {last.date()} y faltan {len(missing)} días hábiles del mes."


def _clean_rows(rows: pd.DataFrame | None) -> list[dict]:
    if rows is None or not len(rows):
        return []
    return json.loads(rows.head(EXAMPLES).astype(str).to_json(orient="records", force_ascii=False))


def _finding(level: str, code: str, message: str, count: int | None = None, rows: pd.DataFrame | None = None) -> dict:
    return {"nivel": level, "codigo": code, "titulo": TITLES.get(code, code), "mensaje": message, "consejo": ADVICE.get(code, ""), "cantidad": count, "ejemplos": _clean_rows(rows)}


def _staging(paths: AppPaths, staging_id: str) -> Path:
    folder = paths.staging / staging_id
    if not staging_id.isalnum() or not folder.is_dir():
        raise LookupError("Esa carga ya no existe. Vuelve a elegir los archivos.")
    return folder


def _sweep(paths: AppPaths) -> None:
    r"""Remove staged uploads nobody finished (older than a day)."""
    if paths.staging.is_dir():
        for folder in paths.staging.iterdir():
            if folder.is_dir() and time.time() - folder.stat().st_mtime > STALE_HOURS * 3600:
                shutil.rmtree(folder, ignore_errors=True)


def stage(paths: AppPaths, files: list[tuple[str, bytes]]) -> dict:
    r"""Save the uploaded exports in a staging folder; only .xlsx files are accepted."""
    if not files:
        raise ValueError("No se eligió ningún archivo.")
    bad = [name for name, _ in files if not name.lower().endswith(".xlsx")]
    if bad:
        raise ValueError(f"Solo se aceptan exportes de Excel (.xlsx). No se puede usar: {', '.join(bad)}.")
    _sweep(paths)
    staging_id = uuid.uuid4().hex[:8]
    folder = paths.staging / staging_id
    folder.mkdir(parents=True)
    for number, (name, content) in enumerate(files, start=1):
        (folder / f"{number:02d}_{Path(name).name}").write_bytes(content)
    return {"id": staging_id, "archivos": [{"nombre": Path(n).name, "bytes": len(c)} for n, c in files]}


def _read_master(paths: AppPaths) -> pd.DataFrame:
    if not paths.consumption_csv.exists():
        return pd.DataFrame({"planta": pd.Series(dtype=str), "sku": pd.Series(dtype=str), "fecha": pd.Series(dtype="datetime64[ns]"), "consumo": pd.Series(dtype=float)})
    # round_trip keeps every digit exactly, so rewriting the master does not change values that were not touched
    return pd.read_csv(paths.consumption_csv, parse_dates=["fecha"], float_precision="round_trip")


def review(paths: AppPaths, staging_id: str, policy: str = "detener", progress: Callable[[str, float], None] | None = None) -> dict:
    r"""Read the staged exports, add them to the current master in memory, check the result and return what the screen shows; writes nothing to the master."""
    if policy not in POLICIES:
        raise ValueError(f"Política desconocida: {policy}.")
    report = progress or (lambda message, fraction: None)
    folder = _staging(paths, staging_id)
    sources = sorted(folder.glob("*.xlsx"))
    parts, files = [], []
    for i, path in enumerate(sources):
        report(f"Leyendo {path.name.split('_', 1)[1]}", 0.05 + 0.5 * i / len(sources))
        try:
            records = read_export(path, drop_zeros=True)
        except Exception:  # openpyxl raises many kinds of errors for files that are not a SAP export
            raise ValueError(f"No se pudo leer {path.name.split('_', 1)[1]}: no parece un exporte de consumo de SAP.") from None
        frame = pd.DataFrame(records, columns=COLUMNS)
        frame["fecha"] = pd.to_datetime(frame["fecha"])
        if frame.empty:
            raise ValueError(f"{path.name.split('_', 1)[1]} no trae filas de consumo.")
        parts.append(frame)
        files.append({"nombre": path.name.split("_", 1)[1], "filas": int(len(frame)), "desde": str(frame["fecha"].min().date()), "hasta": str(frame["fecha"].max().date())})

    report("Comparando con los datos actuales", 0.6)
    master = _read_master(paths)
    new = pd.concat(parts, ignore_index=True)
    grouped = new.groupby(KEYS, as_index=False)["consumo"].agg(consumo="max", minimo="min")
    inside = int(((grouped["consumo"] - grouped["minimo"]).abs() > TOLERANCE).sum())  # the same day twice in the new files with different values
    new = grouped[COLUMNS]
    overlap = master.merge(new, on=KEYS, suffixes=("_actual", "_nuevo"))
    differ = overlap[(overlap["consumo_actual"] - overlap["consumo_nuevo"]).abs() > TOLERANCE]
    conflicts = int(len(differ)) + inside
    kept = master.merge(new[KEYS], on=KEYS, how="left", indicator=True).query("_merge == 'left_only'")[COLUMNS]
    combined = pd.concat([kept, new], ignore_index=True).sort_values(KEYS).reset_index(drop=True)

    report("Revisando los datos", 0.75)
    check = check_consumption(combined)
    findings = [_finding(f.severity, f.code, _spanish(f.code, None if f.rows is None else len(f.rows), f.rows, f.message), None if f.rows is None else len(f.rows), f.rows) for f in check.findings]
    partial = _partial_month(combined)
    if partial:
        findings.append(_finding("warning", "mes_parcial", partial))
    added = len(combined) - len(master)
    same = int(len(overlap) - len(differ))
    if conflicts:
        sample = differ.head(EXAMPLES).rename(columns={"consumo_actual": "actual", "consumo_nuevo": "nuevo"})
        level = "warning" if policy == "usar_nuevo" else "error"
        text = f"{conflicts:,} combinaciones de planta, SKU y día tienen un valor distinto en el exporte nuevo."
        findings.insert(0, _finding(level, "conflictos", text + (" Se usarán los valores del exporte nuevo." if policy == "usar_nuevo" else " Elige qué valor es el correcto."), conflicts, sample))
    elif added == 0:
        findings.append(_finding("error", "sin_cambios", "Todo lo que trae el exporte ya estaba cargado con los mismos valores."))
    errors = [f for f in findings if f["nivel"] == "error"]
    ready = not errors

    months_new = sorted(set(new["fecha"].dt.strftime("%Y-%m")) - set(master["fecha"].dt.strftime("%Y-%m")))
    summary = {
        "filas_actuales": int(len(master)), "filas_nuevas": int(added), "filas_repetidas_iguales": same, "conflictos": conflicts, "filas_resultado": int(len(combined)),
        "meses_nuevos": months_new, "ultimo_dia_actual": None if master.empty else str(master["fecha"].max().date()), "ultimo_dia_resultado": str(combined["fecha"].max().date()),
        "primer_dia_resultado": str(combined["fecha"].min().date()), "plantas": int(combined["planta"].nunique()),
        "plantas_nuevas": sorted(set(new["planta"]) - set(master["planta"])),
    }
    result = {"id": staging_id, "politica": policy, "archivos": files, "resumen": summary, "hallazgos": findings, "puede_aplicar": ready}
    if ready:
        combined.assign(fecha=combined["fecha"].dt.strftime("%Y-%m-%d")).to_csv(folder / "combinado.csv", index=False)
        (folder / "revision.json").write_text(json.dumps({"master_sha": file_sha256(paths.consumption_csv) if paths.consumption_csv.exists() else None, "resultado": result}), encoding="utf-8")
    else:
        (folder / "revision.json").unlink(missing_ok=True)
    report("Listo", 1.0)
    return result


def _prune_backups(paths: AppPaths) -> None:
    backups = sorted(paths.backups.glob("consumos_long_*.csv"))
    for old in backups[:-KEEP_BACKUPS]:
        old.unlink()


def apply(paths: AppPaths, staging_id: str) -> dict:
    r"""Make the reviewed table the master: the previous master is copied to `data/respaldo/` first. Refuses if the master changed after the review."""
    folder = _staging(paths, staging_id)
    review_file, combined_file = folder / "revision.json", folder / "combinado.csv"
    if not review_file.exists() or not combined_file.exists():
        raise ValueError("Esta carga tiene errores o no se ha revisado: no se puede aplicar.")
    saved = json.loads(review_file.read_text(encoding="utf-8"))
    current = file_sha256(paths.consumption_csv) if paths.consumption_csv.exists() else None
    if saved["master_sha"] != current:
        raise ValueError("Los datos cambiaron después de la revisión. Vuelve a revisar los archivos.")
    backup = None
    if paths.consumption_csv.exists():
        paths.backups.mkdir(parents=True, exist_ok=True)
        backup = paths.backups / f"consumos_long_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
        shutil.copy2(paths.consumption_csv, backup)
        _prune_backups(paths)
    table = pd.read_csv(combined_file, float_precision="round_trip")
    write_csv(list(zip(table["planta"], table["sku"], table["fecha"], table["consumo"])), paths.consumption_csv)
    shutil.rmtree(folder, ignore_errors=True)
    summary = saved["resultado"]["resumen"]
    return {"filas_resultado": summary["filas_resultado"], "filas_nuevas": summary["filas_nuevas"], "ultimo_dia": summary["ultimo_dia_resultado"],
            "meses_nuevos": summary["meses_nuevos"], "respaldo": backup.name if backup else None}


def cancel(paths: AppPaths, staging_id: str) -> None:
    shutil.rmtree(_staging(paths, staging_id), ignore_errors=True)


def backups(paths: AppPaths) -> list[dict]:
    r"""The saved copies of the master, newest first."""
    files = sorted(paths.backups.glob("consumos_long_*.csv"), reverse=True) if paths.backups.is_dir() else []
    return [{"archivo": f.name, "creado_en": datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"), "megabytes": round(f.stat().st_size / 1e6, 1)} for f in files]


def undo(paths: AppPaths) -> dict:
    r"""Put the newest backup back as the master (and remove that backup)."""
    saved = backups(paths)
    if not saved:
        raise LookupError("No hay una carga anterior que deshacer.")
    newest = paths.backups / saved[0]["archivo"]
    shutil.copy2(newest, paths.consumption_csv)
    newest.unlink()
    return {"restaurado": saved[0]["archivo"]}
