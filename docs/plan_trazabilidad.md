# Sub-plan 2: trazabilidad de los pronósticos

Rama `feature/run-traceability`. Código: `src/engine/run_manifest.py`, `src/engine/fingerprints.py`, cambios en `scripts/predict_next_month.py`;
pruebas: `tests/test_run_manifest.py` (8).

Junto a cada `results/forecast_<mes>.csv`, `predict_next_month.py` escribe `results/forecast_<mes>.manifest.json` con lo que se usó para hacerlo.
Es un JSON de unos pocos KB; no toca la base de datos. Los datos **no se copian**: el commit y el hash del archivo los identifican cuando el
archivo está en git.

## Contenido del manifiesto

| Bloque | Campos |
|---|---|
| `run` | `run_id` (fecha UTC y hash corto), `created_at`, `forecast_month`, `as_of`, `seconds` |
| `code` | `commit`, `branch`, `dirty` (archivos versionados con cambios sin guardar; se toma antes de la corrida, así que sus propios archivos no cuentan) |
| `data` | `path`, `file_sha256`, `file_in_git`, `file_modified`, `rows`, `rows_in_file`, `first_date`, `last_date`, `n_series`, `panel_fingerprint`, `validation` (errores y avisos de la validación de entrada, con su número de filas) |
| `params` | `tuning_file`, `tuning_file_sha256`, `params_hash`, `tune_window` y `test_window` por modelo, `seed` |
| `environment` | Python, sistema, pandas, numpy, LightGBM, XGBoost, scikit-learn |
| `output` | `csv`, `csv_sha256`, `rows`, `total_ml_mean`, `by_planta` |

## Cómo reconstruir una corrida

1. Abrir el manifiesto y leer `code.commit`, `data.file_sha256` y `params.tuning_file_sha256`.
2. Con `git checkout <commit>` se recupera el código y, si `data.file_in_git` es verdadero y `data.file_modified` falso, el archivo de datos exacto.
3. Si `code.dirty` o `data.file_modified` son verdaderos, el commit no alcanza: había cambios sin guardar. Por eso conviene guardar (hacer commit
   de) el archivo de datos antes de cada corrida importante.
4. Al volver a correr con ese código, datos y parámetros, `output.csv_sha256` debe ser igual.

## Resultados

- Octubre de 2026 regenerado: el CSV quedó idéntico byte a byte; su manifiesto trae `dirty: false`, `file_in_git: true`, `file_modified: false`,
  huella de los datos `729f6f97f581669e`, `params_hash` `bff27ef60dd3ad28` y total 53.219,6 TO.
- `panel_fingerprint` y `params_hash` pasaron a `src/engine/fingerprints.py` (capa `engine`) para que producción no dependa de `evaluation`;
  `cache_utils.py` los sigue exportando, así que ningún import existente cambió.
- Suite completa: 131 pruebas pasan (123 previas y 8 nuevas). Prueba de mutación del constructor: huella de datos, `params_hash`, hash de archivo y
  entorno fijados a una constante hacen fallar al menos una prueba.

## Fuera de este plan
Revisión de la salida, logs, verificador de corridas y tabla `corridas` en SQLite (esta última pasa al sub-plan 5).
