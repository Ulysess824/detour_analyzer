# Codebook de variables del modelo

Cada fila de la tabla de entrenamiento es una serie (planta + SKU) en un mes de origen `o`. El objetivo `y` es el consumo total del mes `t = o + h` (h = 1, el mes siguiente), en toneladas (TO). Las 25 variables usan solo meses `<= o`. Código: `src/utils/feature_utils.py` (`SERIES_FEATURES`).

Notación: `y_m` consumo mensual del SKU en el mes `m`; `a_m` días con consumo de ese mes; `b_m` días hábiles del mes (sin domingos); `Y^P_m` total mensual de la planta del SKU; `Y^G_m` total mensual de todas las plantas; `W_k = {o-k+1, ..., o}` ventana de los últimos `k` meses.

## 1. Tabla de variables

| # | Variable | Grupo | Qué mide | Unidad | Vacía (NaN) cuando |
|---|---|---|---|---|---|
| 1 | `lag1` | Consumo reciente | Consumo del mes de origen | TO | el SKU no existía |
| 2 | `lag2` | Consumo reciente | Consumo de hace 2 meses | TO | el SKU no existía |
| 3 | `lag3` | Consumo reciente | Consumo de hace 3 meses | TO | el SKU no existía |
| 4 | `mean3` | Consumo reciente | Promedio mensual de los últimos 3 meses | TO/mes | no hay datos en la ventana |
| 5 | `mean6` | Consumo reciente | Promedio mensual de los últimos 6 meses | TO/mes | no hay datos en la ventana |
| 6 | `mean12` | Consumo reciente | Promedio mensual de los últimos 12 meses | TO/mes | no hay datos en la ventana |
| 7 | `std6` | Consumo reciente | Variación (desviación estándar) de los últimos 6 meses | TO | no hay datos en la ventana |
| 8 | `rate3` | Ritmo diario | Consumo por día hábil, últimos 3 meses | TO/día | no hay días en la ventana |
| 9 | `rate6` | Ritmo diario | Consumo por día hábil, últimos 6 meses | TO/día | idem |
| 10 | `rate12` | Ritmo diario | Consumo por día hábil, últimos 12 meses | TO/día | idem |
| 11 | `occ3` | Ritmo diario | Proporción de días hábiles con consumo, 3 meses | 0 a 1 | idem |
| 12 | `occ6` | Ritmo diario | Proporción de días hábiles con consumo, 6 meses | 0 a 1 | idem |
| 13 | `occ12` | Ritmo diario | Proporción de días hábiles con consumo, 12 meses | 0 a 1 | idem |
| 14 | `size6` | Ritmo diario | Consumo de un día típico con consumo | TO/día activo | `occ6` es 0 o vacía |
| 15 | `zero_share12` | Intermitencia | Proporción de meses sin consumo en los últimos 12 | 0 a 1 | no hay datos en la ventana |
| 16 | `months_active` | Historia | Meses transcurridos desde que apareció el SKU | meses | nunca |
| 17 | `ly` | Año anterior | Consumo del mismo mes un año antes del objetivo | TO | `t-12 < 0` |
| 18 | `ly_rate` | Año anterior | Lo mismo por día hábil | TO/día | `t-12 < 0` |
| 19 | `ratio_planta` | Estacionalidad | Cuánto pesó el mes objetivo el año anterior en la planta, frente a los 6 meses previos de ese año | razón | `o < 17` |
| 20 | `ratio_global` | Estacionalidad | Lo mismo con el total de todas las plantas | razón | `o < 17` |
| 21 | `lyrel_planta` | Estacionalidad | El mes objetivo del año anterior frente a sus 7 meses vecinos, en la planta | razón | `t-15 < 0` |
| 22 | `lyrel_global` | Estacionalidad | Lo mismo con el total de todas las plantas | razón | `t-15 < 0` |
| 23 | `month_t` | Calendario | Mes del año del objetivo (1 a 12) | mes | nunca |
| 24 | `n_days_t` | Calendario | Días hábiles del mes objetivo (sin domingos) | días | nunca |
| 25 | `planta_code` | Identidad | Código numérico de la planta (0 a 14, orden alfabético) | código | nunca |

Columnas de la tabla que no son variables: `series` (índice de la serie), `o`, `t`, `y` (objetivo) y los pronósticos simples `p_*` (naive, medias, estacional, etc.), que sirven solo como referencia de comparación.

## 2. Cómo se calcula cada variable

Consumo reciente (`m` = mes, `W_k` = últimos `k` meses hasta el origen; se ignoran los meses anteriores a la aparición del SKU):

- `lag1 = y_o`, `lag2 = y_{o-1}`, `lag3 = y_{o-2}`
- `mean_k = (1 / |W_k|) * sum_{j in W_k} y_j`, para `k = 3, 6, 12`
- `std6 = sqrt( (1 / |W_6|) * sum_{j in W_6} (y_j - mean6)^2 )` (desviación poblacional)

Ritmo diario:

- `rate_k = sum_{j in W_k} y_j / sum_{j in W_k} b_j` (TO por día hábil)
- `occ_k = sum_{j in W_k} a_j / sum_{j in W_k} b_j`
- `size6 = rate6 / occ6`

Intermitencia e historia:

- `zero_share12 = (número de meses de W_12 con y_j = 0) / |W_12|`
- `months_active = o - m0 + 1`, con `m0` el primer mes con consumo del SKU

Año anterior y estacionalidad (con `t = o + 1`):

- `ly = y_{t-12}`; `ly_rate = y_{t-12} / b_{t-12}`
- `ratio_P = Y^P_{t-12} / mean( Y^P_{o-17}, ..., Y^P_{o-12} )`; `ratio_G` igual con `Y^G`
- `lyrel_P = Y^P_{t-12} / mean( Y^P_{t-15}, ..., Y^P_{t-9} )`; `lyrel_G` igual con `Y^G`

Calendario e identidad:

- `month_t` = mes del año de `t`; `n_days_t = b_t`; `planta_code` = posición de la planta en la lista alfabética.

### Ejemplo con números reales

SCAN, `K/01/160gsm/2100mm/1200-1450`, origen agosto de 2026, objetivo septiembre de 2026:

| Variable | Cálculo | Valor |
|---|---|---|
| `lag1`, `lag2`, `lag3` | agosto, julio, junio | 96,03 / 100,82 / 96,87 |
| `mean3` | (96,03 + 100,82 + 96,87) / 3 | 97,91 |
| `mean6` | promedio de marzo a agosto (112,5; 93,0; 122,7; 96,9; 100,8; 96,0) | 103,64 |
| `std6` | desviación de esos 6 valores | 10,54 |
| `rate6` | suma de 6 meses 621,87 / 157 días hábiles | 3,961 |
| `occ6` | 95 días con consumo / 157 días hábiles | 0,605 |
| `size6` | 3,961 / 0,605 | 6,546 |
| `zero_share12` | 0 meses en cero de 12 | 0 |
| `months_active` | de 2024-01 a 2026-08 | 32 |
| `ly`, `ly_rate` | septiembre de 2025: 105,54; 105,54 / 26 | 105,54 / 4,059 |
| `ratio_planta` | SCAN sep-2025 (3.671,2) / promedio SCAN mar-ago 2025 (3.696,1) | 0,993 |
| `ratio_global` | total sep-2025 (51.224,1) / promedio mar-ago 2025 (54.791,2) | 0,935 |
| `lyrel_planta` | SCAN sep-2025 (3.671,2) / promedio SCAN jun-dic 2025 (3.608,1) | 1,017 |
| `n_days_t`, `month_t`, `planta_code` | septiembre de 2026; SCAN | 26 / 9 / 7 |

## 3. Cómo las usan los modelos

Los tres modelos de árboles (LightGBM, XGBoost, Random Forest) reciben exactamente estas 25 columnas. LightGBM trata `planta_code` como categórica; Random Forest reemplaza los vacíos por -1. El nombre del SKU nunca entra al modelo.
