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

El CSV maestro tiene 316_576 filas, 15 plantas, 1_047 SKU y cubre 2024-01-02 a 2026-06-30.
Reglas de la transformación:

- `sku` es la descripción del material, no el código numérico.
- Los consumos negativos (reajustes de stock) se llevan a 0.
- Los exportes largos omiten los días sin consumo y el ancho trae ceros explícitos, por eso se
  usa `--drop-zeros`, para que ambos queden con el mismo criterio.

Regenerarlo:

```bash
python scripts/transform_consumos.py --drop-zeros \
    data/consumos_2024_2025.xlsx data/consumos_2026.xlsx -o data/consumos_long.csv
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
    neural_utils.py        DNN y LSTM diarios con pérdida Tweedie
    row_model_utils.py     modelos sobre filas crudas: naive, Croston, hurdle, LightGBM, XGBoost, ARIMA
    row_neural_utils.py    DNN y LSTM sobre filas crudas (estrategia del paper de Ouwehand et al.)
    econometric_utils.py   SES, Holt amortiguado y ARIMA por serie
    ensemble_utils.py      combinaciones: media, mediana, media recortada, ponderada
    mcs_utils.py           Model Confidence Set (Hansen, Lunde y Nason, 2011)
    member_utils.py        pronósticos de todos los miembros de los ensembles
    evaluation_utils.py    evaluación con origen rodante a nivel SKU, planta y total
    search_space_utils.py  espacios de búsqueda de Optuna y trial base
    tuning_utils.py        folds cronológicos, búsqueda, test y resumen
    plot_utils.py          gráfico seaborn de real contra base y Optuna
    viewer_utils.py        visor HTML (plantilla en templates/)
scripts/                   puntos de entrada de línea de comandos
```

## Flujo

Todos los comandos parten del CSV maestro y se ejecutan desde la raíz del repositorio.

```bash
# 1. Análisis exploratorio por planta y SKU
python scripts/eda_consumos.py data/consumos_long.csv

# 2. Comparación de modelos sobre filas crudas (naive, Croston, hurdle, LightGBM, XGBoost, ARIMA)
python scripts/compare_models.py data/consumos_long.csv --cutoff 2026-04-30 [--neural]

# 3. Forecast mensual por SKU, planta y total (origen rodante, h=1 y h=3)
python scripts/forecast_monthly.py data/consumos_long.csv --horizons 1 3 --n-test 12 [--daily]

# 4. Tuning bayesiano (Optuna TPE) de los árboles y de las redes neuronales
python scripts/tune_trees.py data/consumos_long.csv --trials 100 100 40 --out results/tuning_trees.json
python scripts/tune_neural.py data/consumos_long.csv --models dnn --trials 60 --out results/tuning_dnn.json

# 5. Visor interactivo Base vs Optuna (HTML autocontenido)
python scripts/build_tuning_viewer.py results/tuning_trees.json -o results/tuning_viewer.html

# 6. Real vs base vs Optuna en el test (total, o una serie con --planta y --sku)
python scripts/plot_pred_vs_real.py data/consumos_long.csv results/tuning_trees.json -o pred_vs_real.png
python scripts/plot_pred_vs_real.py data/consumos_long.csv results/tuning_trees.json \
    --planta SCAN --sku "TSL/01/80gsm/2450mm/1200-1450" -o pred_scan.png

# 7. Ensembles (machine learning, econometría clásica y ambos) y Model Confidence Set
python scripts/run_ensembles.py data/consumos_long.csv --tuning results/tuning_trees.json
```

## Ensembles y Model Confidence Set

`run_ensembles.py` pronostica el total mensual de cada SKU (origen rodante, h=1) con:

- **Machine learning:** LightGBM, XGBoost y Random Forest con los parámetros de Optuna, y DNN y
  LSTM diarios con la configuración base.
- **Econometría clásica:** naive, medias de 3, 6 y 12 meses, tasa por día, naive estacional, media de
  6 meses por razón estacional de la planta, suavizado exponencial simple, Holt amortiguado y ARIMA
  elegido por AIC.

Cada grupo se combina con cuatro métodos (media, mediana, media recortada y media ponderada por el
inverso del MAE pasado), y un tercer grupo combina los dos anteriores (`ml_econ`). El Model
Confidence Set (pérdida de error absoluto, bootstrap por bloques de meses) indica qué modelos se
pueden descartar con confianza 90% y 75%. Los pronósticos de los miembros se guardan en
`results/member_forecasts.csv` y los resultados en `results/ensembles.json`.

Resultado (test 2025-07 a 2026-06, h=1, confianza 90%):

| nivel | mejor por accuracy | modelos que se pueden descartar |
|---|---|---|
| SKU-mes | LightGBM (71.0%) | 19 de 27, entre ellos XGBoost, DNN, LSTM y todos los modelos clásicos y sus ensembles |
| planta-mes | `ml_econ_mean` (91.2%) | 14 de 27, entre ellos DNN, LSTM, XGBoost y los modelos simples de media |
| total-mes | `ml_econ_mean` (95.6%) | 15 de 27, entre ellos DNN, LSTM, `ml_mean` y `naive` |

Las redes usan la configuración base (sin Optuna), y con solo 12 meses de test el MCS tiene poca
potencia a nivel total.

## Métricas

- **WAPE** = Σ|real − pred| / Σ real.
- **Accuracy** = 1 − WAPE.
- **Bias** = Σ(real − pred) / Σ real. Positivo significa que se predice de menos.
- Se reportan a nivel SKU-mes, planta-mes y total-mes. Al agregar se cancelan errores de signo
  opuesto, por eso la precisión sube con el nivel.

## Resultados hasta ahora (test 2025-07 a 2026-06, h=1)

| nivel | accuracy aprox. |
|---|---|
| SKU-mes | 71% |
| planta-mes | 91% |
| total-mes | 94% |

- El consumo diario es casi ruido alrededor del nivel de cada serie: sin autocorrelación útil. El
  techo viene de los datos, sin variables externas (pedidos, forecast del ERP).
- Tuning con Optuna: la mejora en test solo es creíble para Random Forest. En LightGBM la
  diferencia no se distingue del ruido y XGBoost sobreajusta la validación.
- Resultados completos en `results/tuning_trees.json`.

## Pendiente

- Tuning del grupo de redes neuronales (DNN y LSTM).
- Script de inferencia: entrenar con todo el histórico y predecir el mes siguiente.
