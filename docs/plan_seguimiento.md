# Sub-plan 5: seguimiento contra el real

Rama `feature/tracking`. Código: `src/evaluation/tracking_utils.py`, `scripts/track_forecast.py`, cambios en `src/engine/db_utils.py`,
`scripts/load_predictions.py` y `scripts/build_database.py`; pruebas: `tests/test_tracking.py` (13).

## Flujo

1. **Antes del mes:** `predict_next_month.py` guarda `results/forecast_<mes>.csv` y su manifiesto. Ese archivo es el pronóstico "congelado".
2. **Cuando el mes cierra:** se agregan sus datos al archivo maestro y se corre
   `python scripts/track_forecast.py results/forecast_<mes>.csv --data data/consumos_long.csv`.
3. El script lee el pronóstico y su manifiesto, comprueba que los datos cubren el mes completo (si no, se detiene), calcula el acierto de cada modelo
   en SKU, por planta y total, imprime el resultado y las alertas, y lo guarda en `results/seguimiento.csv` (va a git) y en la tabla `seguimiento` de la base.
   Correrlo otra vez para la misma corrida y mes reemplaza las filas. Termina con código de salida 2 si salta una alerta.

El seguimiento lee el CSV del pronóstico y su manifiesto, no la tabla `consumos_predichos` (que se sobrescribe mientras dure la beta).

## Qué se guarda

`seguimiento` (una fila por corrida, mes, nivel, planta y modelo): `acierto`, `wape`, `bias_pct` (regla del proyecto: positivo = el pronóstico quedó bajo),
`n`, `real_TO`, `pronostico_TO` y `alerta`. Niveles: `sku` (las series pronosticadas juntas), `planta` (todas las plantas juntas, o una por una) y `total`;
más una fila `sin_pronostico` con el volumen de las series que consumieron en el mes pero no tenían pronóstico. Modelos: `ml_mean`, `lgbm`, `xgb`, `rf`, `naive`.
`corridas` guarda una fila por pronóstico (identificador, mes, commit, huella de datos, hash de parámetros, hash del CSV, total) tomada de su manifiesto.
`consumos_predichos` gana la columna `run_id` (la corrida que hizo cada predicción vigente). El esquema pasa a la versión 2 con una migración que solo
agrega (conserva todo lo que ya había). Al reconstruir la base con `build_database.py --rebuild`, el historial vuelve desde `results/seguimiento.csv`.

## Alertas (sobre `ml_mean`; saltan estrictamente por debajo del umbral)

| Código | Umbral | Observado en los 12 meses de backtest |
|---|---|---|
| `sku_bajo` | acierto por SKU menor a 64% | 66,2% a 74,7% |
| `planta_bajo` | acierto por planta (todas juntas) menor a 85% | 87,3% a 95,3% |
| `total_bajo` | acierto del total menor a 90% | 91,2% a 99,6% |
| `peor_que_naive` | `ml_mean` peor que el naive en SKU | nunca |
| `planta_revisar` (informativa) | una planta con acierto menor a 60% | 3 de 180 planta-mes |

Los umbrales se pueden cambiar por línea de comandos (`--sku`, `--planta`, `--total`, `--planta-revisar`).

## Comprobaciones con datos reales

- **Relleno de backtest** (`track_forecast.py --backtest`, corrida `backtest-h1`): 12 meses de `results/member_forecasts.csv`, 1.092 filas (121 KB). Medias
  de `ml_mean`: SKU 70,4%, planta 91,9%, total 97,6%, igual que los resultados vigentes mes a mes, y 0 alertas.
- **De punta a punta:** pronóstico de septiembre de 2026 con `--as-of 2026-08` y seguimiento contra el septiembre real: `ml_mean` 66,2% SKU, 88,4% planta,
  99,2% total (los mismos números del backtest), real 50.555,4 TO contra 50.165,2 TO pronosticados (bias_pct +0,8), 16 series sin pronóstico con 71,1 TO.
  Salió el aviso informativo de SALI (36,4%), coherente con el cambio de nivel de SALI en septiembre que ya estaba anotado.
- Primer mes real de producción: octubre de 2026, cuando cierre.
- 13 pruebas nuevas; prueba de mutación: 8 errores inyectados (alerta en el umbral, comparación con el naive invertida, total sin las series nuevas,
  mes incompleto aceptado, CSV que no reemplaza, base que no borra antes de insertar, migración que no agrega la columna, volumen sin pronóstico mal contado),
  los 8 detectados.
