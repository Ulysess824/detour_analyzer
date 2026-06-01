# detour_analyzer

Sistema de prediccion de consumo diario por planta y SKU con deteccion de picos. Implementa un pipeline de dos etapas: regresion continua (Stage 1) para estimar el volumen de consumo, y clasificacion derivada (Stage 2) para alertar sobre picos comparando la prediccion con un umbral dinamico basado en el forecast mensual.

---

## Objetivo

El problema central es anticipar si el consumo real de un dia va a superar el forecast esperado, lo cual genera desviaciones que afectan la planificacion de inventario. En lugar de atacar el problema directamente como clasificacion binaria, el sistema predice la magnitud continua del consumo (Stage 1) y clasifica el pico de forma dinamica (Stage 2). Esto preserva la señal de magnitud, evita ruido por umbrales artificiales y permite ajustar el criterio de clasificacion sin reentrenar.

**Definicion de pico:**
```
pico = consumo_real > 2.0 * (forecast_mensual / 20)
```

---

## Estructura del proyecto

```
detour_analyzer/
├── src/
│   ├── features/
│   │   └── build_features.py          # Ingenieria de 22+ variables en Polars
│   ├── models/
│   │   ├── lgbm_regressor.py          # Stage 1: LightGBM Regressor (produccion)
│   │   └── tft_regressor.py           # Stage 1: Temporal Fusion Transformer (candidato)
│   ├── optimization/
│   │   └── bayesian_optimizer.py      # BayesianOptimizer — ajuste de hiperparametros
│   └── utils/
│       ├── data_transformation.py     # Transformacion completa del dato crudo
│       ├── regression_metrics.py      # Metricas de regresion con summary statsmodels
│       ├── classification_metrics.py  # Metricas de clasificacion para picos
│       └── plotter.py                 # 10 visualizaciones interactivas con Plotly
├── notebooks/
│   └── 01_Detector_de_Picos.ipynb
├── data/
│   ├── historico_consumo.parquet
│   └── synthetic_consumption.parquet
└── requirements.txt
```

---

## Instalacion

```bash
pip install -r requirements.txt
```

Para usar el TFT se requiere `neuralforecast`:

```bash
pip install "neuralforecast>=1.7.0"
```

---

## Pipeline completo paso a paso

### 1. Transformacion del dato crudo

`transform_data` convierte el DataFrame raw del sistema fuente en un formato analitico listo para `build_features`. Calcula internamente `consumo_real`, `daily_forecast`, los targets `target_consumo_real` y `target_is_peak` para t+1, y añade one-hot encoding de `planta` de forma automatica. No se requiere ninguna preparacion adicional fuera de esta funcion.

```python
import polars as pl
from src.utils import transform_data

df_raw = pl.read_parquet("data/historico_consumo.parquet")

df_preparado = transform_data(
    data=df_raw,
    columnas_para_sku=["Grade", "Sbgr", "Gram", "Width"],
    columna_fecha="fecha_fichero",
    columna_consumo="Stock on hand\n(TO)",
    columna_planta="Customer",
    columna_forecast="Total Fcst M\n(TO)",
    umbral_pico=2.0,
)
```

**Salida:** DataFrame con todas las columnas necesarias para el pipeline:

| Columna | Descripcion |
|---|---|
| `planta`, `sku`, `fecha` | Identificadores de serie |
| `consumo_real` | Consumo diario calculado (fill_null de consumo_anterior_to) |
| `consumo_anterior_to` | Diferencia de stock entre dias consecutivos |
| `stock_planta` | Stock disponible en planta |
| `forecast_mensual` | Forecast mensual del ERP |
| `daily_forecast` | forecast_mensual / 20 |
| `target_consumo_real` | consumo_real del dia t+1 (target de regresion) |
| `target_is_peak` | 1 si el dia t+1 es pico, 0 si no (target de clasificacion) |
| `planta_{X}` | One-hot encoding de cada planta (generado automaticamente) |

---

### 2. Ingenieria de variables

`build_features` construye 22+ variables predictoras en Polars. `sku_stats` se calcula solo sobre entrenamiento para evitar data leakage en validacion y test.

```python
from src.features import build_features

# Separacion cronologica — nunca aleatoria para series temporales
fecha_corte = pl.date(2025, 10, 31)
df_train_raw = df_preparado.filter(pl.col("fecha") <= fecha_corte)
df_test_raw  = df_preparado.filter(pl.col("fecha") >  fecha_corte)

# Estadisticas de SKU calculadas solo sobre train
df_train, sku_stats = build_features(df_train_raw)

# Las mismas estadisticas se aplican al test (sin recalcular)
df_test, _ = build_features(df_test_raw, sku_stats=sku_stats)
```

**Variables generadas:**

| Grupo | Variables | Cantidad |
|---|---|---|
| Lags de consumo | `lag_1`, `lag_2`, `lag_3`, `lag_5`, `lag_10` | 5 |
| Ventanas moviles | `rolling_mean/std/max` para ventanas de 3, 5 y 10 dias | 9 |
| Calendario | `day_of_month`, `days_to_end_of_month`, dias de semana, meses | 21 |
| Desvio acumulado | `cum_actual_consumption_month_lag1`, `cum_forecast_month_lag1`, `cum_deviation_month_lag1`, `daily_forecast` | 4 |
| Perfil historico SKU | `sku_historical_peak_rate`, `sku_historical_cv` | 2 |

Todas las ventanas moviles y lags se calculan sobre `consumo_real_lag1` (valor del dia anterior) para evitar lookahead bias.

---

### 3a. Modelo de produccion: LightGBM Regressor

`LGBMRegressor` envuelve a `lightgbm.LGBMRegressor` con una interfaz de pipeline que integra `RegressionMetrics` y `ConsumptionPlotter`. El modelo es global: se entrena sobre todas las plantas y SKUs simultaneamente. La identidad de cada serie queda codificada en las features `sku_historical_peak_rate`, `sku_historical_cv` y el one-hot de `planta`.

```python
from src.models import LGBMRegressor

features_base = [
    "lag_1", "lag_2", "lag_3", "lag_5", "lag_10",
    "rolling_mean_3", "rolling_std_3", "rolling_max_3",
    "rolling_mean_5", "rolling_std_5", "rolling_max_5",
    "rolling_mean_10", "rolling_std_10", "rolling_max_10",
    "day_of_month", "days_to_end_of_month",
    "daily_forecast",
    "cum_actual_consumption_month_lag1",
    "cum_forecast_month_lag1",
    "cum_deviation_month_lag1",
    "sku_historical_peak_rate",
    "sku_historical_cv",
]
features_calendario = [c for c in df_train.columns if c.startswith("day_of_week_") or c.startswith("month_")]
features_planta     = [c for c in df_train.columns if c.startswith("planta_")]
feature_cols = features_base + features_calendario + features_planta

model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
model.fit(df_train, x_predictor=feature_cols, y="target_consumo_real")
print(model.summary())

# Stage 1: prediccion continua
pred_consumo = model.predict(df_test)

# Stage 2: clasificacion derivada de picos
umbral      = 2.0 * df_test["daily_forecast"].to_numpy()
pred_pico   = (pred_consumo > umbral).astype(int)

fig = model.plot_fit(df_test, plant="SCVA", sku="HP1_01_190_2290")
fig.show()
```

**Propiedades disponibles tras `fit()`:**

```python
model.n_obs          # numero de observaciones de entrenamiento
model.train_mae      # MAE sobre train
model.train_rmse     # RMSE sobre train
model.r2             # R-squared sobre train
model.adj_r2         # R-squared ajustado sobre train
model.train_metrics  # objeto RegressionMetrics completo
```

---

### 3b. Modelo candidato: Temporal Fusion Transformer

`TFTRegressor` envuelve a `neuralforecast.models.TFT` de Nixtla — el modelo no se construye desde cero. El TFT implementa redes de seleccion de variables, atencion multi-cabeza sobre el historico temporal y encoders separados para covariables futuras y features estaticos por serie.

```python
from src.models import TFTRegressor

model = TFTRegressor(
    h=1,
    input_size=15,
    hidden_size=64,
    n_heads=4,
    n_lstm_layers=2,
    dropout=0.1,
    max_steps=500,
    batch_size=32,
    early_stop_patience_steps=10,
    futr_exog_list=["daily_forecast", "day_of_month", "days_to_end_of_month"],
    stat_exog_list=["sku_historical_peak_rate", "sku_historical_cv"],
    freq="D",
)

model.fit(df_train, y="consumo_real", date_col="fecha",
          group_cols=["planta", "sku"], val_size=20)

df_preds    = model.predict_dataframe(df_test)   # DataFrame con pred_consumo
pred_consumo = model.predict(df_test)            # np.ndarray

model.save("models/tft_checkpoint/")
model_loaded = TFTRegressor.load("models/tft_checkpoint/")
```

**Regla de seleccion:**

| Condicion | Modelo recomendado |
|---|---|
| SKU con < 90 dias de historia | `LGBMRegressor` |
| Serie con >= 90 dias de historia | `TFTRegressor` |
| Se necesita interpretabilidad inmediata | `LGBMRegressor` + SHAP |
| Se necesita intervalo de confianza | `LGBMRegressor` con quantile loss |
| Se sospecha shift de distribucion | Revisar Adversarial AUC antes de elegir |

---

### 4. Optimizacion de hiperparametros

`BayesianOptimizer` aplica optimizacion bayesiana con el sampler TPE de Optuna. Es completamente agnnostico al modelo — funciona con cualquier clase que implemente `__init__(**hyperparams)`, `fit(df, **fit_kwargs)` y `predict(df) -> np.ndarray`. Usa walk-forward CV expandible internamente, nunca splits aleatorios.

```python
from src.optimization import BayesianOptimizer
from src.models import LGBMRegressor

def lgbm_space(trial):
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1_000),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

opt = BayesianOptimizer(
    model_class=LGBMRegressor,
    param_space_fn=lgbm_space,
    fit_kwargs={"x_predictor": feature_cols, "y": "target_consumo_real"},
    metric="mae",       # "mae" | "wape" | "rmse"
    n_trials=50,
    n_cv_folds=3,       # walk-forward expanding CV
    val_size=30,        # dias de validacion por fold
)

study = opt.optimize(df_train)

print(f"Best MAE:    {opt.best_value():.4f}")
print(f"Best params: {opt.best_params()}")

# Reentrenar el mejor modelo en todo el train
best_model = opt.best_model(df_train)
pred = best_model.predict(df_test)

# Resultados de todos los trials
df_trials = opt.results_dataframe()
```

**Visualizaciones del optimizador:**

```python
opt.plot_optimization_history().show()    # convergencia por trial
opt.plot_param_importance().show()        # que hiperparametros importan mas (fANOVA)
opt.plot_parallel_coordinates().show()    # que regiones del espacio dan mejores resultados
```

**Para TFT** se define otro `param_space_fn` y se pasa `model_class=TFTRegressor` — el resto del codigo es identico.

---

### 5. Evaluacion de modelos

Todas las evaluaciones usan **walk-forward cross-validation** (nunca split aleatorio).

```python
from src.utils import RegressionMetrics, ClassificationMetrics

# Stage 1 — Regresion
reg = RegressionMetrics(model_name="LGBMRegressor", n_features=len(feature_cols))
reg.compute(
    y_true=df_test["target_consumo_real"].to_numpy(),
    y_pred=pred_consumo,
    y_baseline=df_test["lag_1"].to_numpy(),   # baseline lag-1 obligatorio
)
print(reg.summary())

# Stage 2 — Clasificacion derivada
scores = pred_consumo / np.where(umbral == 0, 1.0, umbral)
clf = ClassificationMetrics(model_name="LGBMRegressor", beta=2.0)
optimal_threshold = clf.find_optimal_threshold(y_true_class, scores, metric="fbeta")
clf.compute(y_true=y_true_class, y_pred_proba=scores, threshold=optimal_threshold)
print(clf.summary())
```

**Metricas de regresion (Stage 1):**

| Metrica | Descripcion | Objetivo |
|---|---|---|
| MAE | Error absoluto medio | Referencia en unidades del negocio |
| RMSE | Raiz del error cuadratico medio | <= 1.2 * MAE |
| WAPE | Error porcentual absoluto ponderado | < 0.20 |
| Bias | Media del residuo | Cercano a 0 |
| MAE Lift | Mejora porcentual vs baseline lag-1 | > 0% obligatorio |

**Metricas de clasificacion (Stage 2):**

| Metrica | Descripcion |
|---|---|
| Recall | Prioritario: perder un pico real cuesta mas que una falsa alarma |
| F-beta (beta=2) | Penaliza mas los falsos negativos que los falsos positivos |
| PR-AUC | Preferido sobre ROC-AUC bajo desequilibrio de clases |
| KS Statistic | Separacion entre distribuciones de pico y no-pico |
| Adversarial AUC | Detecta shift entre train y test; objetivo: cerca de 0.5 |

---

### 6. Visualizacion

`ConsumptionPlotter` centraliza las 10 visualizaciones del sistema. Todos los metodos devuelven `go.Figure` y aceptan `save_path` para exportar HTML.

```python
from src.utils import ConsumptionPlotter

plotter = ConsumptionPlotter(model_name="LGBMRegressor")
df_res_pd = df_resultado.to_pandas()
```

**Diagnostico actual vs predicho:**

```python
# Lineas actual vs predicho con TP/FP/FN/TN por punto
plotter.plot_forecast(df_res_pd, date_col="fecha", actual_col="target_consumo_real",
                      pred_col="pred_consumo", threshold_col="umbral_pico",
                      plant="SCVA", sku="HP1_01_190_2290").show()

# Scatter: sesgo sistematico y varianza — puntos coloreados por |error|
plotter.plot_scatter_actual_vs_pred(
    y_true=y_true_real, y_pred=y_pred_real,
    dates=df_resultado["fecha"].to_numpy(),
).show()

# Error absoluto en el tiempo vs MAE del modelo y MAE del baseline lag-1
plotter.plot_error_over_time(
    y_true=y_true_real, y_pred=y_pred_real,
    dates=df_resultado["fecha"].to_numpy(),
    baseline_mae=reg.baseline_mae,
).show()

# Grid de todas las series ordenadas por MAE descendente
plotter.plot_forecast_grid(df_res_pd, date_col="fecha",
                           actual_col="target_consumo_real",
                           pred_col="pred_consumo", max_series=12).show()
```

**Diagnostico de residuos y Stage 2:**

```python
plotter.plot_residuals(y_true_real, y_pred_real, dates=fechas).show()
plotter.plot_pr_curve(y_true_class, scores, pr_auc=clf.pr_auc).show()
plotter.plot_roc_curve(y_true_class, scores, roc_auc=clf.roc_auc).show()
plotter.plot_calibration(y_true_class, scores, n_bins=10).show()
```

**SHAP e interpretabilidad:**

```python
import shap
explainer   = shap.TreeExplainer(model.model)
shap_values = explainer.shap_values(df_test_pd[feature_cols])
plotter.plot_shap_importance(shap_values, feature_names=feature_cols, top_n=20).show()
```

**Comparacion entre modelos:**

```python
metricas = [lgbm_model.train_metrics.to_dict(), tft_model.val_metrics.to_dict()]
plotter.plot_model_comparison(metricas, metric_cols=["mae", "rmse", "wape"]).show()
```

**Catalogo completo de metodos:**

| Metodo | Proposito |
|---|---|
| `plot_forecast` | Actual vs predicho en el tiempo con overlay TP/FP/FN/TN |
| `plot_scatter_actual_vs_pred` | Scatter Y=X con coloreado por error — detecta sesgo |
| `plot_error_over_time` | Error absoluto diario vs MAE modelo y baseline |
| `plot_forecast_grid` | Subplots por serie ordenados por MAE — identifica series problematicas |
| `plot_residuals` | Panel de 3: residuos en el tiempo, distribucion, residuos vs fitted |
| `plot_shap_importance` | Importancia de features por media de |SHAP| |
| `plot_pr_curve` | Curva Precision-Recall con area bajo la curva |
| `plot_roc_curve` | Curva ROC con AUC |
| `plot_calibration` | Diagrama de fiabilidad (calibracion del score) |
| `plot_model_comparison` | Barras agrupadas comparando metricas entre modelos |

---

## Estandares de codigo

- Lenguaje: ingles en codigo y comentarios inline.
- Python 3.10+. Separador de miles con guion bajo (`1_000`).
- Toda modificacion a una clase requiere ejemplo de uso en chat.
- Validacion cronologica obligatoria: nunca `train_test_split` aleatorio en series temporales.
- Todo modelo debe superar el baseline de persistencia lag-1 en MAE y WAPE.
