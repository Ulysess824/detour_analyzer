# detour_analyzer

Forecast del consumo por planta y SKU a partir de exportes de SAP. El objetivo es predecir la serie
(no los picos): consumo mensual por SKU, por planta y total, agregando a partir de series por SKU.

## Datos

| archivo | contenido |
|---|---|
| `data/consumos_2026.xlsx` | exporte en formato ancho (hoja `RESULT`), 2026 |
| `data/consumos_2024_2025.xlsx` | exporte en formato largo, 2024-2025 |
| `data/consumos_2024_ene.xlsx` | subconjunto duplicado de enero 2024; no se usa |
| `data/consumos_long.csv` | **CSV maestro**: `planta, sku, fecha, consumo` |

El CSV maestro tiene 347_502 filas, 15 plantas y cubre 2024-01-02 a 2026-09-30 (33 meses).
Reglas de la transformación:

- `sku` es la descripción del material, no el código numérico.
- Los consumos negativos (reajustes de stock) se llevan a 0.
- Los exportes largos omiten los días sin consumo y el ancho trae ceros explícitos, por eso se
  usa `--drop-zeros`, para que ambos queden con el mismo criterio.

Regenerarlo:

```bash
python scripts/transform_consumos.py --drop-zeros \
    data/consumos_2024_2025.xlsx data/consumos_2025_10_2026_09.xlsx -o data/consumos_long.csv
```

## Instalación

```bash
pip install -r requirements.txt
```

## Estructura

```
src/utils/                 funciones reutilizables, una responsabilidad por módulo
    transform_utils.py     lectura de los exportes SAP (ancho y largo) y escritura del CSV
    eda_utils.py           estadísticas descriptivas por serie y por planta
    panel_utils.py         panel mensual (Panel, build_panel) y ventanas móviles
    feature_utils.py       features, baselines simples y make_frame
    daily_utils.py         expansión de filas mensuales a días calendario
    metrics_utils.py       WAPE, accuracy, bias y tablas con ranking
    tree_utils.py          LightGBM, XGBoost y Random Forest (mensual y diario)
    econometric_utils.py   SES, Holt amortiguado y ARIMA por serie
    ensemble_utils.py      combinaciones: media, mediana, media recortada, ponderada
    mcs_utils.py           Model Confidence Set (Hansen, Lunde y Nason, 2011)
    member_utils.py        pronósticos de todos los miembros de los ensembles
    evaluation_utils.py    evaluación con origen rodante a nivel SKU, planta y total
    search_space_utils.py  espacios de búsqueda de Optuna y trial base
    tuning_utils.py        folds cronológicos, búsqueda, test y resumen
    cache_utils.py         protecciones de los resultados guardados (huella de datos, alineación, ventana de ajuste)
    planner_io_utils.py    lectura de los libros de asignación del planificador (todas las plantas)
    internal_io_utils.py   lectura del libro del modelo interno (todas las plantas), reglas de faltantes
    excel_utils.py         el cruce de predicciones como libro de Excel: hoja de comparación y hoja de excluidos
    planner_utils.py       emparejamiento y comparación con el pronóstico del planificador
    inference_utils.py     pronóstico del mes siguiente con todo el histórico
    horizon_utils.py       pronóstico a dos meses: directa, iterada y con mes parcial
    partial_utils.py       variables del mes en curso (consumo de los primeros días)
scripts/                   puntos de entrada de línea de comandos
```

## Flujo

Todos los comandos parten del CSV maestro y se ejecutan desde la raíz del repositorio.

```bash
# 1. Análisis exploratorio por planta y SKU
python scripts/eda_consumos.py data/consumos_long.csv

# 2. Forecast mensual por SKU, planta y total (origen rodante, h=1 y h=3)
python scripts/forecast_monthly.py data/consumos_long.csv --horizons 1 3 --n-test 12 [--daily]

# 3. Tuning bayesiano (Optuna TPE) de LightGBM, XGBoost y Random Forest
python scripts/tune_trees.py data/consumos_long.csv --trials 100 100 40 --out results/tuning_trees.json

# 4. Ensembles (machine learning, econometría clásica y ambos) y Model Confidence Set
python scripts/run_ensembles.py data/consumos_long.csv --tuning results/tuning_trees.json

# 5. Pronóstico del mes siguiente con todo el histórico (con --as-of YYYY-MM se valida contra el real conocido)
python scripts/predict_next_month.py data/consumos_long.csv

# 6. Pronóstico a dos meses (plazo del planificador): directa, parcial e iterada, más la fila del planificador
python scripts/compare_horizon_strategies.py data/consumos_long.csv

# 7. Pronóstico del planificador contra el modelo: convertir los libros del planificador y comparar (septiembre, VMI, un mes)
python scripts/transform_planner.py Asignacion_jun_26.xlsx Asignacion_jul_26.xlsx Asignacion_ago_26.xlsx Asignacion_SEPT_26.xlsx --out-dir data
python scripts/compare_planner.py data/consumos_long.csv --months 2026-09

# 8. Con el modelo interno: convertir su libro y comparar solo los SKU que están en las tres fuentes
python scripts/transform_internal.py proposed_forecast_internal_model_all_plants.xlsx
python scripts/compare_planner.py data/consumos_long.csv --months 2026-09 --internal data/internal_forecast_plantas.csv

# 9. Excel con dos tablas: predicciones (real, planificador, modelo interno, ml_mean) y excluidos con su motivo
python scripts/export_comparison_excel.py --month 2026-09
```

## Ensembles y Model Confidence Set

`run_ensembles.py` pronostica el total mensual de cada SKU (origen rodante, h=1) con:

- **Machine learning:** LightGBM, XGBoost y Random Forest con los parámetros de Optuna.
- **Econometría clásica:** naive, medias de 3, 6 y 12 meses, tasa por día, naive estacional, media de
  6 meses por razón estacional de la planta, suavizado exponencial simple, Holt amortiguado y ARIMA
  elegido por AIC.

Cada grupo se combina con cuatro métodos (media, mediana, media recortada y media ponderada por el
inverso del MAE pasado), y un tercer grupo combina los dos anteriores (`ml_econ`). El Model
Confidence Set (pérdida de error absoluto, bootstrap por bloques de meses) indica qué modelos se
pueden descartar con confianza 90% y 75%. Los pronósticos de los miembros se guardan en
`results/member_forecasts.csv` y los resultados en `results/ensembles.json`.

Resultado (test 2025-10 a 2026-09, h=1, confianza 90%, 25 modelos: 3 de machine learning, 10
clásicos y 12 ensembles):

| nivel | mejor por accuracy | modelos que se pueden descartar |
|---|---|---|
| SKU-mes | `ml_mean` (70.4%) | 21 de 25: todos los clásicos, XGBoost, LightGBM y casi todos los ensembles |
| planta-mes | `ml_median` (91.9%) | 14 de 25: XGBoost, ARIMA, SES, Holt y las medias simples |
| total-mes | `ml_mean` (97.5%) | 8 de 25: XGBoost, ARIMA, `naive` y las medias simples |

Con 3 miembros de machine learning la media recortada (20%) no recorta ninguno, así que
`ml_trimmed` es igual a `ml_mean`. Con solo 12 meses de test el MCS tiene poca potencia a nivel
total.

## Features de los modelos de machine learning

LightGBM, XGBoost y Random Forest pronostican **el total mensual de cada SKU** (una fila por serie y
por origen, no por día ni por semana) con los mismos 25 features. Se definen en
`src/utils/feature_utils.py` (`SERIES_FEATURES`) y se construyen con `make_frame(S, horizonte)`.

Notación: el **origen** `o` es el último mes con datos conocidos, el **mes objetivo** es `t = o + h`
(con `h = 1`, el mes siguiente) y `y` es el consumo total del mes objetivo, que es lo que se predice.
`monthly[m]` es el consumo total de la serie en el mes `m`; `bd[m]` son los días hábiles del mes `m`
(días del mes sin domingos).

| # | feature | grupo | qué mide | cálculo |
|---|---|---|---|---|
| 1 | `lag1` | rezagos | consumo del último mes observado | `monthly[o]` |
| 2 | `lag2` | rezagos | consumo de hace dos meses | `monthly[o-1]` |
| 3 | `lag3` | rezagos | consumo de hace tres meses | `monthly[o-2]` |
| 4 | `mean3` | ventanas | nivel reciente | promedio de `o-2` a `o` |
| 5 | `mean6` | ventanas | nivel de medio plazo | promedio de `o-5` a `o` |
| 6 | `mean12` | ventanas | nivel de largo plazo | promedio de `o-11` a `o` |
| 7 | `std6` | ventanas | volatilidad reciente | desviación estándar de `o-5` a `o` |
| 8 | `zero_share12` | intermitencia | proporción de meses sin consumo | meses con consumo 0 / meses vividos, de `o-11` a `o` |
| 9 | `months_active` | antigüedad | edad de la serie | `o - primer_mes + 1` |
| 10 | `ly` | año anterior | consumo del mismo mes un año antes | `monthly[t-12]` |
| 11 | `ly_rate` | año anterior | lo anterior por día hábil | `monthly[t-12] / bd[t-12]` |
| 12 | `planta_code` | identidad | planta de la serie | código entero, orden alfabético |
| 13 | `month_t` | calendario | mes del año del mes objetivo | 1 a 12 |
| 14 | `n_days_t` | calendario | días hábiles del mes objetivo | `bd[t]` |
| 15 | `ratio_planta` | estacionalidad | cómo se comportó la planta ese mes el año pasado, contra sus 6 meses previos | `planta[t-12] / promedio(planta[o-17..o-12])` |
| 16 | `ratio_global` | estacionalidad | lo mismo para el total de todas las plantas | `total[t-12] / promedio(total[o-17..o-12])` |
| 17 | `lyrel_planta` | estacionalidad | el mes del año pasado contra su entorno de 7 meses | `planta[t-12] / promedio(planta[t-15..t-9])` |
| 18 | `lyrel_global` | estacionalidad | lo mismo para el total | `total[t-12] / promedio(total[t-15..t-9])` |
| 19 | `rate3` | intensidad | consumo por día hábil, 3 meses | suma de `o-2..o` / días hábiles de esos meses |
| 20 | `rate6` | intensidad | consumo por día hábil, 6 meses | igual con `o-5..o` |
| 21 | `rate12` | intensidad | consumo por día hábil, 12 meses | igual con `o-11..o` |
| 22 | `occ3` | ocurrencia | fracción de días hábiles con consumo, 3 meses | días con consumo / días hábiles, `o-2..o` |
| 23 | `occ6` | ocurrencia | igual, 6 meses | `o-5..o` |
| 24 | `occ12` | ocurrencia | igual, 12 meses | `o-11..o` |
| 25 | `size6` | tamaño | consumo medio por día con consumo, 6 meses | `rate6 / occ6` |

### Explicación de cada grupo

**Rezagos (`lag1`, `lag2`, `lag3`).** Son los totales mensuales más recientes de la propia serie.
`lag1` es el mes inmediatamente anterior al que se pronostica (con `h = 1`) y suele ser el
predictor individual más fuerte: es el pronóstico "naive". Tener también `lag2` y `lag3` le deja
al modelo ver la **dirección** (sube, baja o se mantiene) y distinguir un mes atípico de un cambio
de nivel.

**Ventanas móviles (`mean3`, `mean6`, `mean12`, `std6`).** Los promedios resumen el nivel de la serie
en tres horizontes. `mean3` reacciona rápido a cambios, `mean12` es estable y casi no se mueve con
un mes raro, y `mean6` queda entre ambos. Comparar uno con otro da una noción de tendencia: si
`mean3 > mean12`, la serie viene subiendo. `std6` mide cuánto oscila el consumo mensual alrededor de
su nivel; una serie con `std6` alto es difícil de pronosticar y el modelo aprende a no fiarse tanto
de `lag1`. Los promedios ignoran los meses anteriores a la primera aparición de la serie (no los
cuentan como ceros) pero sí cuentan los ceros posteriores.

**Intermitencia y antigüedad (`zero_share12`, `months_active`).** Muchos SKU no se consumen todos los
meses. `zero_share12` es la proporción de meses de los últimos 12 con consumo exactamente cero (solo
cuenta los meses desde que la serie existe): 0 es una serie continua, valores cercanos a 1 son una
serie casi inactiva. Le dice al modelo si debe esperar ceros. `months_active` es la edad de la serie
en meses: una serie de 4 meses tiene un `mean12` poco fiable y una de 24 meses ya tiene historia
estacional. Además separa los SKU recientes de los antiguos.

**Año anterior (`ly`, `ly_rate`).** `ly` es el consumo del **mismo mes calendario** un año antes del
mes objetivo (para pronosticar junio de 2026, el consumo de junio de 2025). Captura estacionalidad
propia del SKU (campañas, cierres anuales). `ly_rate` es lo mismo dividido por los días hábiles de
aquel mes, para que el modelo pueda separar "se consumió más" de "el mes tenía más días". Con menos
de 12 meses de historia son nulos.

**Identidad y calendario (`planta_code`, `month_t`, `n_days_t`).** `planta_code` identifica la planta
(código entero por orden alfabético); en LightGBM se declara categórico, mientras que en XGBoost y
Random Forest entra como número. `month_t` es el mes del año (1 a 12) del mes que se predice, y deja
al modelo aprender estacionalidad común entre SKU. `n_days_t` es el número de días hábiles del mes
objetivo (sin domingos): un mes con 26 días hábiles consume más que uno con 24, y este feature
corrige ese efecto de calendario que de otro modo se confundiría con ruido.

**Estacionalidad agregada (`ratio_planta`, `ratio_global`, `lyrel_planta`, `lyrel_global`).** La
estacionalidad de un SKU individual es demasiado ruidosa para estimarla, así que se mide sobre
agregados: la planta (`_planta`) y el total de todas las plantas (`_global`). Por eso valen lo mismo
para todos los SKU de la misma planta (o, en `_global`, para todos los SKU del origen).
- `ratio_*` responde: *el año pasado, ese mes, ¿cómo estuvo el consumo frente a los seis meses
  anteriores a ese origen?* Es el consumo agregado del mes `t-12` dividido entre el promedio de los
  meses `o-17` a `o-12`, que es la misma ventana de 6 meses que usa `mean6` pero un año antes. Un
  valor de 1,2 significa que el mes objetivo estuvo un 20% por encima de la media reciente de ese
  año; un valor de 0,8, un 20% por debajo.
- `lyrel_*` mide lo mismo con otro denominador: el promedio de los 7 meses **centrados** en el mes
  del año pasado (`t-15` a `t-9`). Aísla el efecto "ese mes es alto o bajo respecto de sus vecinos"
  de la tendencia general, que `ratio_*` mezcla con la estacionalidad.

  Ejemplo ilustrativo (valores inventados): se pronostica junio de 2026 desde mayo de 2026. Si una
  planta consumió 120 en junio de 2025 y el promedio de diciembre de 2024 a mayo de 2025 fue 100,
  entonces `ratio_planta = 1,2`. Si el promedio de marzo a septiembre de 2025 fue 110,
  `lyrel_planta = 120 / 110 = 1,09`.

Los `ratio_*` necesitan que existan los meses `t-12` y `o-17`, y los `lyrel_*` que exista `t-15`, así
que son nulos en los primeros orígenes de la historia.

**Intensidad por día hábil (`rate3`, `rate6`, `rate12`).** Es el consumo acumulado de la ventana
dividido entre los días hábiles de los meses en que la serie ya existía. Como el consumo mensual
depende de cuántos días tiene el mes, la tasa por día es una medida de nivel más comparable entre
meses que el promedio mensual. El modelo puede multiplicarla mentalmente por `n_days_t` para obtener
un pronóstico ajustado a calendario (es la lógica del modelo clásico `perday6`).

**Ocurrencia (`occ3`, `occ6`, `occ12`).** Es la fracción de días hábiles en que hubo consumo
positivo: días con consumo / días hábiles de la ventana. Cercana a 1, el SKU se consume casi a
diario; cercana a 0, es esporádico. Como cuenta días y no volumen, permite distinguir una serie
que consume poco todos los días de otra que consume mucho unos pocos días al año. Puede superar
ligeramente 1 si hubo consumo en domingo, porque el numerador cuenta cualquier día con consumo y el
denominador excluye los domingos.

**Tamaño (`size6`).** `rate6 / occ6` equivale al consumo medio **por día con consumo** en los
últimos 6 meses. Junto con `occ*` descompone la demanda en dos factores: *con qué frecuencia* se
consume (`occ`) y *cuánto* se consume cada vez (`size6`). Es nulo si no hubo ningún día con
consumo en la ventana.

### Reglas que valen para todos los features

- **Sin fuga de datos.** Todas las ventanas terminan en el origen `o` o antes. Los datos del año
  anterior (`t-12`, `o-17` a `o-12` y `t-15` a `t-9`) ya estaban observados en el origen para
  `h = 1` y `h = 3`. Nada del mes objetivo entra como feature.
- **Series vivas.** Solo hay fila para las series que ya existían en el origen, por eso los SKU que
  aparecen después (arranque en frío) no se pronostican.
- **Valores nulos.** Aparecen cuando la historia no alcanza (por ejemplo `ly` cuando el mes `t-12`
  cae antes del inicio de los datos o la serie aún no existía, o los rezagos en series muy nuevas). LightGBM y XGBoost los manejan
  directamente; Random Forest los rellena con -1.
- **Las columnas `p_*` del frame no son features.** `make_frame` guarda junto a los features los
  pronósticos de los modelos simples (`p_naive`, `p_mean3`, `p_mean6`, ...), pero se usan solo como
  baselines y miembros de los ensembles, nunca como entrada de los modelos de machine learning.
- **Modelo diario.** `forecast_monthly.py --daily` usa estos mismos 25 features más cuatro de día
  (`dow`, `dom`, `day_idx`, `days_left`), predice cada día calendario del mes y suma. No forma parte
  de los ensembles.
- **Modelos clásicos.** Los miembros clásicos de los ensembles (naive, medias, SES, Holt, ARIMA) no
  usan estos features, solo la historia mensual de cada serie.

## Métricas

- **WAPE** = Σ|real − pred| / Σ real.
- **Accuracy** = 1 − WAPE.
- **Bias** = Σ(real − pred) / Σ real. Positivo significa que se predice de menos.
- Se reportan a nivel SKU-mes, planta-mes y total-mes. Al agregar se cancelan errores de signo
  opuesto, por eso la precisión sube con el nivel.

## Resultados hasta ahora (test 2025-10 a 2026-09, h=1)

| nivel | accuracy aprox. |
|---|---|
| SKU-mes | 70% |
| planta-mes | 92% |
| total-mes | 97% |

- El consumo diario es casi ruido alrededor del nivel de cada serie: sin autocorrelación útil. El
  techo viene de los datos, sin variables externas (pedidos, forecast del ERP).
- Tuning con Optuna: la mejora en test solo es creíble para Random Forest. En LightGBM la
  diferencia no se distingue del ruido y XGBoost sobreajusta la validación.
- Resultados completos en `results/tuning_trees.json`.

## Pendiente

- Pronóstico de producción a dos meses (estrategia principal y secundaria): `predict_next_month.py` hoy pronostica el mes
  siguiente (h=1). Falta una opción `--horizon 2` y la versión con mes parcial, que necesita un exporte de SAP a mitad de mes.
- Evaluación con un rango de meses como test: hoy la prueba es un origen móvil a un mes (cada mes de prueba se pronostica con los
  datos hasta el mes anterior, reentrenando cada vez; 12 pruebas de octubre de 2025 a septiembre de 2026). Se mantiene esa metodología.
  A futuro conviene valorar además un corte único con un rango de meses como test (entrenar hasta una fecha y pronosticar varios
  meses seguidos), que mide el pronóstico a varios horizontes desde un mismo punto. Hay que cuidar que las variables de los meses
  del rango no usen el consumo real de meses anteriores del propio rango, porque eso sería pronóstico a un mes y no a varios. Los
  resultados a dos y tres meses de hoy (`forecast_monthly.py --horizons`, `compare_horizon_strategies.py`) usan origen móvil por horizonte.
- Varias plantas en las comparaciones: `compare_planner.py` ya compara las 11 plantas del planificador y guarda la planta en cada fila, y el
  cruce con el planificador de `compare_internal_model.py` usa planta, mes y SKU. Falta que los meses del modelo interno, fijos en
  `src/utils/internal_utils.py` (2026-06 a 2026-09), salgan del propio archivo, y la pestaña PLANIFICADOR del dashboard sigue leyendo solo los
  `results/planner_comparison_<mes>.csv` de SCAN (no hay selector de planta). Si cambia `data/consumos_long.csv`, las cachés piden `--refit`.
- Subgrado distinto en el archivo del planificador de septiembre de 2026: 16 filas VMI (SALI 8 y SALM 8, `KS/01/215gsm/<ancho>mm`) aparecen
  con subgrado 01, pero en `data/consumos_long.csv` esos SKU existen como `KS/257/215gsm/<ancho>mm`. No se cruzan y se ignoran en la comparación
  con el planificador (1.484 TO, 3,5% del volumen VMI de septiembre). Pendiente: confirmar con el planificador si es el mismo producto y, si lo es,
  enlazarlos (por ejemplo ignorando el subgrado cuando hay un único candidato en la planta).
  Evidencia de que es el mismo producto: los códigos SAP de esas filas (por ejemplo 1454942, 1519037, 1520783) aparecen como `KS/257/215gsm` en el
  archivo de junio y como `KS/01/215gsm` en el de septiembre. El código SAP serviría para enlazarlos sin adivinar. `compare_planner.py` ya las
  deja fuera con el motivo "subgrado distinto" en `results/filas_excluidas_planificador.csv`.
- Modelo interno: no se sabe en qué fecha se generó cada pronóstico, así que no se puede descartar que use información posterior a la que
  tenía el planificador. Conviene preguntarlo a quien lo envió.
- Rama `low_history`: sigue sin fusionar a `main` por decisión del usuario. Tiene la prueba para SKU con poca historia (1 a 3 meses) y la versión
  de la presentación con el modelo directo a dos meses.
- Presentación: la versión LaTeX (`presentacion/presentacion.tex`) está desactualizada frente a la presentación en vivo
  (`presentacion/artifact/`): no tiene las diapositivas de método, medida, variables ni modelo interno. Las cifras de las diapositivas se
  escriben a mano a partir de `results/internal_model_comparison.csv`.

## Comparación con el planificador

**Datos.** Los archivos de asignación del planificador (un libro por mes, con la planta en `Customer`) se convierten con
`scripts/transform_planner.py` a `data/planner_forecast_<mes>.csv`: junio a septiembre de 2026, 11 plantas, unas 770 filas por mes y todas las
estrategias (VMI, NO VMI, VMI EST). El mes se lee de la cabecera de la hoja y la columna `allocation` (la fábrica que produce) se descarta.
Las filas de SCAN de junio a agosto coinciden con las de antes; septiembre pasa de 32 a 45 SKU de SCAN (los 13 que faltaban eran NO VMI).

**Comparación.** `scripts/compare_planner.py data/consumos_long.csv --months 2026-09` compara el pronóstico del planificador con el de los
modelos sobre los mismos SKU y el mismo mes. El SKU del planificador no trae el ancho de núcleo, así que se suman las series del dataset con
igual planta, tipo, gramaje y ancho. Parámetros: `--months`, `--plantas` (por defecto todas), `--strategies` (por defecto solo `VMI`; agregar
NO VMI es `--strategies VMI "NO VMI"`, sin tocar código) y `--min-history` (por defecto 3). Resultado por mes en `results/planner_plantas_<mes>.csv`.
Usa el modelo con datos hasta fin del mes anterior, que son más de los que tiene el planificador a mitad de mes; la comparación en igualdad de
condiciones está en la sección siguiente.

**Qué se deja fuera, y dónde queda registrado.**

| Motivo | Archivo |
|---|---|
| Menos de 3 meses de historia antes del mes que se pronostica (el primer consumo cuenta como mes 1) | `results/skus_poca_historia.csv`: planta, SKU y primera fecha con consumo |
| El SKU no existe en los datos (nunca tuvo consumo) | `results/skus_poca_historia.csv` (sin fecha) y `results/filas_excluidas_planificador.csv` |
| Subgrado distinto entre el archivo y los datos (ver Pendiente) | `results/filas_excluidas_planificador.csv` |
| El modelo no tiene pronóstico para el SKU | `results/filas_excluidas_planificador.csv` |

Septiembre de 2026, solo VMI, un mes de horizonte: 463 filas en los archivos, 443 comparadas, 16 fuera por subgrado y 4 por poca historia.

| Planta | Filas | Real (TO) | Planificador (TO) | Acierto planificador | Acierto `ml_mean` |
|---|---|---|---|---|---|
| PCEL | 20 | 1.856 | 2.055 | 68,4% | 71,7% |
| SALC | 36 | 4.284 | 4.491 | 82,2% | 83,9% |
| SALI | 52 | 2.559 | 3.855 | 44,9% | 35,2% |
| SALM | 39 | 3.111 | 3.080 | 76,1% | 55,4% |
| SBUR | 54 | 5.505 | 5.745 | 81,6% | 81,1% |
| SCAN | 32 | 3.169 | 3.391 | 80,6% | 86,1% |
| SCOC | 45 | 4.161 | 3.913 | 79,7% | 81,4% |
| SCVA | 44 | 2.968 | 3.133 | 80,0% | 76,8% |
| SPAL | 43 | 2.920 | 3.221 | 69,9% | 77,9% |
| SQUA | 52 | 4.503 | 4.705 | 78,4% | 79,2% |
| SVIG | 26 | 2.473 | 2.515 | 85,6% | 82,6% |
| **Total** | **443** | **37.509** | **40.104** | **76,6%** | **75,4%** |

En las 11 plantas el planificador queda 1,2 puntos por encima del modelo, y `ml_mean` queda más cerca del real que el planificador en 233 de
443 SKU-mes. En SCAN el modelo gana (86,1% contra 80,6%), pero la diferencia total la deciden SALI y SALM. El modelo pierde por mucho
en SALM (55,4% contra 76,1%), y en SALI los dos fallan (el planificador pronostica 3.855 TO y el real fue 2.559). Esas plantas cambiaron de
nivel: el consumo de SALI cae de 6.381 TO en agosto a 3.653 TO en septiembre (17 días con consumo, contra 21 a 26 en las otras plantas del planificador), el de SALM
pasa de 7.173 TO en mayo a 4.867 TO en junio, y el de SPAL baja en julio. Un modelo que aprende del pasado no anticipa esos cambios.
Pendiente de revisar con quien entrega los datos si son cambios reales o exportaciones incompletas.
Un mes y 11 plantas son una referencia, no una prueba.

**Con el modelo interno.** `scripts/transform_internal.py` convierte el libro del modelo interno (11 plantas, junio a septiembre) a
`data/internal_forecast_plantas.csv`. Dos reglas: SCAN viene en cero en el libro, así que se toman sus valores del archivo anterior
(`data/internal_model_forecast.csv`, 42 SKU), y un SKU con 0 en los cuatro meses se escribe como faltante (vacío), porque un cero exacto
en todos los meses indica que el modelo no dio pronóstico (43 SKU). Con `--internal`, `compare_planner.py` calcula todas las métricas solo
con los SKU que están en el planificador (VMI), en el modelo interno y en la comparación de `ml_mean`. Los SKU sin pronóstico del modelo
interno quedan en la tabla guardada con `internal_status = faltante` y fuera de las métricas.
Septiembre de 2026, VMI: 443 SKU comparados con `ml_mean`, 407 con las tres fuentes, 36 con el modelo interno faltante.

**Cómo se cruzan las fuentes.** Planificador y modelo interno se cruzan por planta y nombre del SKU, que se arma con las columnas del libro
(tipo, subgrado, gramaje y ancho: `K/01/110gsm/2100mm`); el código SAP viaja en las tablas pero no se usa para cruzar. En septiembre, por
nombre y por código se encuentran los mismos 445 SKU VMI y ningún par tiene nombres o códigos distintos (en junio 4 SKU `KW2` tienen el mismo
nombre y distinto código). `ml_mean` se cruza también por planta y nombre: los datos de consumo no traen código SAP, solo la descripción del
material (`K/01/110gsm/2100mm/1200-1450`), y se usan sus cuatro primeras partes, sumando las series que difieren solo en el ancho de núcleo.
`scripts/export_comparison_excel.py` escribe `results/comparacion_<mes>.xlsx` con dos tablas: Comparación (SKU, planta, mes, real y los tres
pronósticos, con "faltante" donde el modelo interno no tiene) y Excluidos (cada fila fuera de las métricas con su motivo).

| | Planificador | `ml_mean` | Modelo interno |
|---|---|---|---|
| Acierto (407 SKU) | 77,2% | 77,2% | 62,8% |
| Más cerca del real (de 407; un empate no cuenta para nadie) | 138 | 171 | 95 |

El modelo interno no tiene sesgo (pronóstico total igual al real, -0,2%) pero se equivoca mucho SKU por SKU (WAPE 37%). `ml_mean` y el
planificador quedan empatados en las 11 plantas. Los 16 `KS/01/215gsm` del pendiente de subgrado no están en el archivo del modelo
interno (ni como 01 ni como 257), así que no entrarían a este conjunto aunque se enlazaran.

## Pronóstico a dos meses: estrategia principal y secundaria

**Problema.** El planificador envía a mitad de mes el pronóstico del mes siguiente (a mitad de octubre, noviembre). En ese momento
se tiene el último mes completo (septiembre) y un mes en curso incompleto. El modelo del resto del README pronostica con datos
hasta fin del mes anterior, que son más de los que tiene el planificador. Para comparar en igualdad de condiciones y pronosticar
noviembre hay que pronosticar **dos meses** desde el último mes completo (`t = o + 2`).

**Decisión.**

| | Estrategia | Qué es |
|---|---|---|
| Principal | **Directa** | `ml_mean` (promedio de LightGBM, XGBoost y Random Forest) entrenado con el objetivo a dos meses. Mismas 25 variables, calculadas en el origen `o`; se entrena con las filas cuyo objetivo ya se conoce (`t <= o`) |
| Secundaria | **Directa + mes parcial** | La directa más 4 variables del mes en curso (`mtd`, `mtd_rate`, `mtd_occ`, `mtd_proj`: consumo de los primeros 15 días, por día hábil, proporción de días con consumo y proyección del mes). Requiere un exporte de SAP a mitad de mes |

La iterada (un modelo a un mes aplicado dos veces) y los modelos locales por SKU (SES, Holt, ARIMA; opción `--with-local`) se
midieron pero no se adoptan.

**De dónde sale.** La literatura describe las alternativas; la elección se hizo con la prueba de este proyecto (origen rodante,
12 meses), porque las fuentes no coinciden entre sí:

| Tema | Fuente | Qué aporta aquí |
|---|---|---|
| Estrategias a varios pasos (directa, iterada, de múltiples salidas) | Ben Taieb, Bontempi, Atiya y Sorjamaa (2012), *Expert Systems with Applications* 39(8): [A review and comparison of strategies for multi-step ahead time series forecasting](https://research.monash.edu/en/publications/a-review-and-comparison-of-strategies-for-multi-step-ahead-time-s/) | Define las estrategias que se comparan. En su prueba (111 series) ganan las de múltiples salidas; esa estrategia no se probó aquí |
| Directa contra iterada | Marcellino, Stock y Watson (2006), *Journal of Econometrics* 135: [A comparison of direct and iterated multistep AR methods](https://cadmus.eui.eu/handle/1814/42713) | En teoría la iterada es más eficiente si el modelo está bien especificado y la directa es más robusta a errores de especificación; en sus datos macroeconómicos gana la iterada. Como no hay consenso, se midió |
| Información parcial del período en curso | Giannone, Reichlin y Small (2008), *Journal of Monetary Economics*: [Nowcasting: the real-time informational content of macroeconomic data](https://lbsresearch.london.edu/id/eprint/315) | Idea de actualizar el pronóstico con datos incompletos del período actual: origen de la estrategia secundaria |
| Un modelo global para todas las series | Montero-Manso y Hyndman (2021), *International Journal of Forecasting* 37(4): [Principles and algorithms for forecasting groups of time series](https://arxiv.org/pdf/2008.00444) | Respalda entrenar un solo modelo con todos los SKU en lugar de uno por serie |
| Combinar modelo y criterio experto | Blattberg y Hoch (1990), *Management Science* 36(8): [Database models and managerial intuition](https://ideas.repec.org/a/inm/ormnsc/v36y1990i8p887-899.html) | Queda aparte: la mezcla con el pronóstico del planificador no se implementó |

**Resultado** (`scripts/compare_horizon_strategies.py`, test 2025-10 a 2026-09, 24.407 SKU-mes; acierto = 1 − WAPE):

| Estrategia | SKU | Planta | Total |
|---|---|---|---|
| Referencia a un mes (más información) | 71,0% | 91,8% | 97,5% |
| **Directa (principal)** | 66,2% | 90,0% | 94,1% |
| Iterada | 65,4% | 89,8% | 94,4% |
| **Directa + mes parcial (secundaria)** | 68,7% | 90,9% | 96,0% |
| Naive (mes anterior) | 62,1% | 85,6% | 90,6% |

- La directa es igual o mejor que la iterada a nivel SKU en los 12 meses de prueba.
- El mes parcial mejora a la directa en los 12 meses a nivel SKU (+2,6 puntos en promedio).

**Contra el planificador** (SCAN, 167 SKU-mes, junio a septiembre de 2026, `results/horizon_planner_comparison.csv`):

| | Acierto | Filas más cerca del real que el planificador |
|---|---|---|
| Planificador | 83,0% | |
| Referencia a un mes | 85,2% | 95 de 167 |
| Directa | 84,3% | 88 |
| Directa + mes parcial | 84,1% | 91 |
| Iterada | 84,1% | 84 |

En igualdad de condiciones la ventaja sobre el planificador es de 1 a 1,3 puntos, y en agosto de 2026 el planificador gana a las
tres estrategias. En los SKU grandes y estables de SCAN el mes parcial no mejora a la directa; la mejora viene de los SKU
pequeños o intermitentes.

**Límites.** Son 12 meses de prueba, y para SCAN cuatro meses y una planta. Los parámetros de Optuna son los ajustados a un mes.
El corte al día 15 se simula con los datos diarios históricos. Cada pronóstico del mes `t` usa solo lo conocido al fin del mes
`t − 2` (y, en la secundaria, los primeros días de `t − 1`); hay una prueba que lo verifica.

## Hiperparámetros: cuándo se ajustan

Los árboles de `ml_mean` (LightGBM, XGBoost y Random Forest) se **reentrenan cada mes** con todo el historial disponible, pero sus
**hiperparámetros se ajustan una sola vez** (Optuna, `scripts/tune_trees.py`, ventana 2024-01 a 2025-09, para pronosticar un mes
adelante; resultado en `results/tuning_trees.json`, unos 20 minutos). `ml_mean` es un promedio simple y no tiene parámetros propios.

Criterio de reajuste (práctico, no sale de una fuente): cada 6 a 12 meses; cuando el WAPE de los últimos meses suba de forma
sostenida; cuando cambien los datos (plantas o SKU nuevos, otro registro del consumo); y antes de usar el pronóstico a dos meses,
porque los parámetros actuales son los de un mes. La ganancia del ajuste fue pequeña (solo Random Forest mejoró de forma creíble),
así que reajustar cada mes no se justifica. `check_tuning_window` impide evaluar con parámetros cuya ventana de ajuste toque los
meses de prueba.

## Modelo interno de la empresa (SCAN)

`data/internal_model_forecast.csv` es el pronóstico propuesto por el modelo interno de la empresa para junio a septiembre de 2026
(45 SKU de SCAN; el SKU se toma del código SAP, porque el archivo no trae subtipo y tres productos THP aparecen como TSL; tres
productos TWTC2 no existen en los datos y quedan fuera). `scripts/compare_internal_model.py` lo compara con el real, con el
planificador y con nuestro modelo sobre los mismos SKU-mes (156 con los tres pronósticos), con el modelo a un mes (la estrategia de la presentación);
la versión directa a dos meses está en la rama `low_history`.

| Acierto, 156 SKU-mes, jun a sep 2026 | |
|---|---|
| Modelo, a un mes | 85,3% |
| Planificador | 83,2% |
| Naive (mes anterior) | 81,0% |
| Modelo interno | 76,7% |

El modelo interno pronostica de más (+7,3% sobre el real, el planificador +3,2% y el modelo -0,4%) y queda por debajo del naive.
El modelo quedó más cerca del real en 69 de 156 SKU-mes, el planificador en 50 y el modelo interno en 37. Los resultados por fila
están en `results/internal_model_comparison.csv`. La presentación (`presentacion/artifact/`) usa estas cifras.

## Pruebas

```bash
python -m pytest -q
```

Usan datos sintéticos y tardan unos 15 segundos (43 pruebas). Cubren: ausencia de fuga en los features, sumas del panel y de la expansión diaria,
pesos de los ensembles, pronóstico del mes siguiente y a dos meses (sin mirar el futuro), reconciliación, signo del bias, folds de Optuna, protecciones de caché y de ventana de ajuste,
reglas de solape al combinar exportes, emparejamiento de SKU del planificador, determinismo de los árboles y el MCS.
Los resultados guardados llevan metadatos (`results/member_forecasts.meta.json` y `meta` dentro de `results/tuning_trees.json`):
si cambian los datos o la ventana de test, los scripts se detienen y piden recalcular (`--refit` o `tune_trees.py`).
