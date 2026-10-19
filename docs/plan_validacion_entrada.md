# Sub-plan 1: validación de entrada

Rama `feature/input-validation`. Código: `src/engine/data_checks.py`, `scripts/validate_input.py`; pruebas: `tests/test_data_checks.py` (22).
`load_consumption(path, validate=True)` valida por defecto en todos los scripts; `validate=False` es la salida explícita.

El validador nunca corrige: detiene o avisa. La limpieza sigue en `scripts/transform_consumos.py`. Los datos entran en toneladas, por eso no hay un
segundo tope contra errores de unidad.

## Reglas

Errores (detienen la carga con `InputValidationError`, que es un `SystemExit`):

| Código | Regla |
|---|---|
| `columns` | Están `planta`, `sku`, `fecha`, `consumo` |
| `types` | `fecha` es fecha, `consumo` es número, `planta` y `sku` son texto |
| `missing` | Sin valores vacíos ni nombres en blanco |
| `negative` | Sin consumos negativos |
| `duplicate` | Sin repetidos de (planta, sku, fecha); se listan las dos copias |
| `keys` | Planta en mayúsculas y sin espacios; SKU sin espacios al inicio o al final |
| `near_duplicate_keys` | Dos series que solo difieren en mayúsculas o espacios |
| `month_gap` | Todos los meses entre la primera y la última fecha tienen filas |
| `day_gap` | Sin 3 o más días hábiles seguidos sin ninguna fila |
| `last_month` | El último mes está completo; solo con `require_complete_last_month=True` (lo usa `validate_input.py`; `predict_next_month.py` mantiene su propio chequeo, porque con `--as-of` el último mes del archivo puede estar incompleto) |

Avisos (la carga sigue y se imprimen las filas, por stderr):

| Código | Regla | Estado en el maestro real |
|---|---|---|
| `daily_peak` | Consumo de un SKU en una planta en un día mayor a 100 TO; muestra planta, SKU, día y consumo | 1 caso: MMRO, `TW2000/51/120gsm/1800mm/1200-1450`, 2024-05-03, 141,7 TO |
| `spike` | Valor diario mayor a 20 veces la mediana de su serie | el mismo caso (36,7 veces) |
| `rows_drop` | Las filas del mes caen 40% o más frente al mes anterior | ninguno |
| `planta_set` | Una planta aparece o desaparece en el último mes frente a los meses anteriores | ninguno |
| `sku_format` | SKU fuera del formato `grado/subgrado/gsm/ancho/…` | `SF1/200GSM/2100MM` |

## Tablas (`python scripts/validate_input.py <archivo>`, formato largo, en `results/`)

| Archivo | Contenido | Filas en el maestro |
|---|---|---|
| `validacion_domingos.csv` | `planta`, `sku`, `fecha`, `consumo` de cada fila en domingo; cuentan como consumo real | 673 (1.993,6 TO) |
| `validacion_picos.csv` | Mismas columnas, para los días con más de 100 TO | 1 |
| `validacion_cambio_planta_sku.csv` | Una fila por mes, planta y SKU: `consumo_mes_anterior`, `consumo_mes`, `cambio_TO`, `cambio_pct`, `estado` (`nuevo`, `sin consumo`, `cambia`) | 37.697 |
| `validacion_cambio_planta.csv` | Una fila por mes y planta, con las mismas columnas y `skus_con_consumo` | 480 (15 plantas por 32 meses) |

- El primer mes no aparece (no tiene mes anterior). Se usan meses de calendario: un mes sin datos figura con consumo 0.
- `cambio_pct` queda vacío cuando el mes anterior fue 0 (`nuevo`).
- La suma del cambio de todos los SKU de una planta es igual al cambio de la planta (prueba).
- El cambio de nivel no es un aviso: ya se conoce y solo se informa en estas tablas.

## Resultados con los datos reales

- Maestro (347.502 filas): 0 errores y 3 avisos (el pico de MMRO, el mismo caso como `spike` y el SKU `SF1/200GSM/2100MM`).
- Tiempo: 0,69 s para las reglas y 0,75 s con las tablas (lectura del CSV aparte: 0,54 s).
- Los casos que antes fallaban en silencio ahora se detienen: 1.000 duplicados, planta en minúscula, SKU con espacio, 50 negativos, 50 vacíos,
  un mes faltante (mayo de 2026) y una columna faltante. Un valor de 1e9 TO ya no detiene (decisión: solo aviso) y sale como `daily_peak` y `spike`.
- El pronóstico de octubre no cambió (diferencia 0,0 TO, total 53.219,6 TO).
- Prueba de mutación: quitar cada una de las 12 reglas, la estructura, `missing` o las tablas de cambio hace fallar al menos una prueba (15 de 15).
- Suite completa: 123 pruebas pasan (101 previas y 22 nuevas).

## Integración

- `load_consumption` (csv o `.db`) valida por defecto; el cargador no escribe archivos.
- `write_consumption` (SQLite) rechaza los datos con errores (`ValueError`) con las mismas reglas.
- `inference_utils.check_last_month_complete` usa el mismo mensaje (`incomplete_last_month_message`).
- `SKU_PATTERN` vive en `data_checks.py` y `db_utils.py` lo importa.
