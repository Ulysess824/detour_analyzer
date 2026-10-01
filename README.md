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

## Flujo

Todos los comandos parten del CSV maestro.

```bash
# 1. Análisis exploratorio por planta y SKU
python scripts/eda_consumos.py data/consumos_long.csv

# 2. Comparación de modelos diarios (naive, Croston, hurdle, LightGBM, XGBoost, ARIMA)
python scripts/compare_models.py data/consumos_long.csv --cutoff 2026-04-30 [--neural]

# 3. Forecast mensual por SKU, planta y total (origen rodante, h=1 y h=3)
python scripts/forecast_monthly.py data/consumos_long.csv --horizons 1 3 --n-test 12 [--daily]

# 4. Tuning bayesiano (Optuna TPE) de LightGBM, XGBoost y Random Forest
python scripts/tune_trees.py data/consumos_long.csv --trials 100 100 40 --out results/tuning_trees.json

# 5. Visor interactivo Base vs Optuna (HTML autocontenido)
python scripts/build_tuning_viewer.py results/tuning_trees.json -o results/tuning_viewer.html

# 6. Real vs base vs Optuna en el test (total, o una serie con --planta y --sku)
python scripts/plot_pred_vs_real.py data/consumos_long.csv results/tuning_trees.json -o pred_vs_real.png
python scripts/plot_pred_vs_real.py data/consumos_long.csv results/tuning_trees.json \
    --planta SCAN --sku "TSL/01/80gsm/2450mm/1200-1450" -o pred_scan.png
```

## Scripts

| script | función |
|---|---|
| `transform_consumos.py` | convierte exportes SAP (ancho o largo) a CSV; acepta varios archivos |
| `eda_consumos.py` | análisis exploratorio por planta y SKU |
| `compare_models.py` | modelos diarios y ranking; `--neural` añade DNN y LSTM (`neural_models.py`) |
| `forecast_monthly.py` | núcleo: panel mensual, features, LightGBM Tweedie, métricas por nivel, reconciliación |
| `daily_model.py` | predice cada día del mes objetivo y luego suma (con objetivo diario rellenado con ceros) |
| `tune_trees.py` | búsqueda Optuna con folds cronológicos y test sin tocar |
| `build_tuning_viewer.py` | genera el visor HTML a partir del JSON de resultados |
| `plot_pred_vs_real.py` | gráfico seaborn de real contra base y Optuna |

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
