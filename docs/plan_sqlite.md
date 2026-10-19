# Base de datos SQLite: consumos reales y consumos predichos (fase 1)

Estado: implementado en la rama `feature/sqlite`. El CSV `data/consumos_long.csv` sigue siendo el maestro; la base se genera desde él
y no se sube a git (`data/*.db`, datos confidenciales y archivo binario).

## Alcance

- Entra: consumos reales y consumos predichos (cuatro tablas).
- No entra: planificador, modelo interno, comparaciones, dashboard, cachés con sus guardas (`cache_utils`). Siguen como estaban.

## Esquema

| Tabla | Contenido | Clave |
|---|---|---|
| `plantas` | 15 plantas | `planta_id`, `codigo` único |
| `skus` | 1.072 descripciones de material, separadas en grade, subgrade, gsm, ancho_mm, ancho_core, sufijo y `clave_planificador` | `sku_id`, `descripcion` única |
| `consumos` | consumo real diario en TO (347.502 filas; 0 solo se omite, nunca negativo) | (`planta_id`, `sku_id`, `fecha`) |
| `consumos_predichos` | predicción mensual en TO | (`planta_id`, `sku_id`, `mes_objetivo`, `mes_origen`, `estrategia`, `modelo`) |

- La serie del proyecto es el par (planta, sku): una descripción se repite entre plantas (1.072 SKU, 2.247 series).
- Vista `consumo_mensual`: consumo real sumado por planta, sku y mes, con los días con consumo.
- Restricciones: claves foráneas activas, `consumo >= 0`, `valor >= 0`, fechas en formato `YYYY-MM-DD`, y `mes_origen < mes_objetivo`.
- El horizonte no se guarda: es `mes_objetivo` menos `mes_origen`.
- Estrategias: `h1` (un mes adelante), `directa`, `parcial`, `iterada` (dos meses adelante) y `nowcast_media` (mes en curso completado con la media).
- Modelos: `lgbm`, `xgb`, `rf`, `ml_mean`, `naive`.

## Regla de beta: las predicciones se sobrescriben

- Cada carga borra y vuelve a escribir el bloque (`estrategia`, `mes_origen`) completo, en una sola transacción. Si algo falla a la mitad, queda
  la versión anterior; si un SKU desaparece de la nueva corrida, no queda una fila vieja.
- Cada fila lleva `creado_en` (fecha y hora UTC de la carga).
- Cuando termine la beta: agregar una columna `corrida` a la clave primaria para conservar el historial.

## Código

| Pieza | Qué hace |
|---|---|
| `src/engine/db_utils.py` | esquema, `write_consumption`, `read_consumption`, `replace_predictions`, `read_predictions`, `parse_sku`, conversión de cachés a formato largo |
| `scripts/build_database.py` | crea la base desde el CSV y comprueba que el panel leído de la base es igual al del CSV |
| `scripts/load_predictions.py` | carga las predicciones guardadas en `results/` (revisa antes `check_alignment` de cada caché) |
| `load_consumption` (panel_utils) | lee también un archivo `.db` o `.sqlite` |
| `tests/test_database.py` | 11 pruebas: restricciones, ida y vuelta, sobrescritura por bloque, reversa ante fallo, traducción del índice de series |

## Comprobaciones con los datos reales

- 15 plantas, 1.072 SKU y 347.502 filas de consumo; huella del panel y suma por planta y mes idénticas a las del CSV.
- Predicciones cargadas (340.383 filas): `h1` 135.030, `directa` 122.035, `iterada` 24.407, `parcial` 24.407, `nowcast_media` 34.504.
- Acierto por SKU recalculado con SQL sobre la base: `ml_mean` `h1` 70,4% y naive 67,0% (24.759 SKU-mes), `parcial` 68,7%, `nowcast_media` 65,4%; coincide con los resultados vigentes.
- Total de octubre de 2026 (`ml_mean`, `h1`): 53.220 TO, igual que `results/forecast_2026-10.csv`.
- Cargar dos veces seguidas deja el mismo número de filas (no duplica).

## Uso

```bash
python scripts/build_database.py data/consumos_long.csv --db data/detour.db --rebuild
python scripts/load_predictions.py data/consumos_long.csv --db data/detour.db
```

## Consultas útiles

```sql
-- consumo mensual de un SKU en una planta
SELECT m.mes, m.dias_con_consumo, ROUND(m.consumo, 3) AS to_
FROM consumo_mensual m JOIN plantas p USING (planta_id) JOIN skus s USING (sku_id)
WHERE p.codigo = 'SCAN' AND s.descripcion = 'HP2/01/125gsm/2440mm/1200-1450' AND m.mes >= '2026-01'
ORDER BY m.mes;

-- predicción de octubre por planta
SELECT p.codigo, ROUND(SUM(q.valor)) AS to_
FROM consumos_predichos q JOIN plantas p USING (planta_id)
WHERE q.mes_objetivo = '2026-10' AND q.estrategia = 'h1' AND q.modelo = 'ml_mean'
GROUP BY p.codigo ORDER BY to_ DESC;
```

## Pendiente

- Fase 2: planificador, modelo interno y comparaciones en la base; que las cachés y el dashboard lean de ella.
- Al terminar la beta: historial de predicciones (`corrida`).
