# detour_analyzer — Context for Claude Code

## What this project does

Two-stage daily consumption prediction system per plant and SKU:
- **Stage 1 (Regression)**: predicts the continuous consumption magnitude for day t+1.
- **Stage 2 (Derived Classification)**: classifies a peak by comparing the Stage 1 output against a dynamic threshold (`2.0 * daily_forecast`). No separate classifier is trained — the threshold is applied post-hoc.

Peak definition: `consumo_real > 2.0 * (forecast_mensual / 20)`

---

## Coding standards (always apply)

- All code and inline comments in **English**.
- Python 3.10+. Use `|` union types, not `Optional`.
- No emojis anywhere — not in print statements, docstrings, or markdown.
- Thousands separator with underscore: `1_000`, not `1000`.
- Every class/method modification requires a usage example in chat and a synthetic-data test script.
- Never use random train/test splits — always **chronological walk-forward** splits.
- Every model must beat the lag-1 persistence baseline on MAE and WAPE.

---

## Directory structure

```
src/
  features/
    build_features.py        # build_features(df, sku_stats=None) -> (df, sku_stats)
  models/
    lgbm_regressor.py        # LGBMRegressor   — production tabular model
    tft_regressor.py         # TFTRegressor    — neuralforecast TFT wrapper
    lstm_model.py            # HybridLSTM, ConsumptionDataset (legacy classifier)
  utils/
    data_transformation.py   # transform_data(data, ...) -> pl.DataFrame
    data_generation.py       # generate_data() -> pl.DataFrame
    selection.py             # select_random_group(df) -> (plant, sku)
    regression_metrics.py    # RegressionMetrics dataclass
    classification_metrics.py# ClassificationMetrics dataclass
    plotter.py               # ConsumptionPlotter (Plotly)

scripts/
  generate_synthetic_data.py
  train_tabular.py
  train_lstm.py
  run_pipeline_demo.py

notebooks/
  01_Detector_de_Picos.ipynb

data/                        # gitignored — parquet files
plots/                       # gitignored — generated outputs
```

---

## Key classes and interfaces

### `transform_data` — src/utils/data_transformation.py
Converts raw ERP export to analysis-ready DataFrame.
```python
df = transform_data(
    data=df_raw,
    columnas_para_sku=["Grade", "Sbgr", "Gram", "Width"],
    columna_fecha="fecha_fichero",
    columna_consumo="Stock on hand\n(TO)",
    columna_planta="Customer",
    columna_forecast="Total Fcst M\n(TO)",
    umbral_pico=2.0,
)
# Output columns: planta, sku, fecha, consumo_anterior_to, forecast_mensual, target_is_peak
```

### `build_features` — src/features/build_features.py
Builds 22 engineered features in Polars. Always compute `sku_stats` on training data only, then pass it to test to avoid leakage.
```python
df_train, sku_stats = build_features(df_train_raw)
df_test, _          = build_features(df_test_raw, sku_stats=sku_stats)
```
Features: 5 lags, 9 rolling windows (mean/std/max over 3/5/10d), 21 calendar (day_of_month, days_to_end_of_month, weekday dummies, month dummies), 4 cumulative deviation features, 2 SKU profile features (peak_rate, cv).

### `LGBMRegressor` — src/models/lgbm_regressor.py
Wraps `lightgbm.LGBMRegressor`. Internally uses `_LGBMRegressor` alias to avoid name collision.
```python
model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
model.fit(df_train, x_predictor=feature_cols, y="consumo_real")
model.predict(df_test)          # -> np.ndarray
model.summary()                 # prints RegressionMetrics + Gain-based feature importance
model.plot_fit(df_test, plant=..., sku=..., save_path=...)  # -> go.Figure (Plotly)

# Properties (read from train_metrics after fit):
model.n_obs, model.train_mae, model.train_rmse, model.r2, model.adj_r2
model.train_metrics             # RegressionMetrics object
```

### `TFTRegressor` — src/models/tft_regressor.py
Wraps `neuralforecast.models.TFT`. Model is imported, not built from scratch.
```python
model = TFTRegressor(
    h=1, input_size=15, hidden_size=64, n_heads=4, n_lstm_layers=2,
    futr_exog_list=["daily_forecast", "day_of_month", "days_to_end_of_month"],
    stat_exog_list=["sku_historical_peak_rate", "sku_historical_cv"],
)
model.fit(df_train, y="consumo_real", date_col="fecha",
          group_cols=["planta", "sku"], val_size=20)
model.predict_dataframe(df_test)   # -> DataFrame with pred_consumo column
model.predict(df_test)             # -> np.ndarray
model.summary()
model.plot_fit(df_test, plant=..., sku=...)
model.save("models/tft/")
TFTRegressor.load("models/tft/")
```
neuralforecast requires `unique_id`, `ds`, `y` format — conversion is handled internally.

### `RegressionMetrics` — src/utils/regression_metrics.py
```python
m = RegressionMetrics(model_name="LGBMRegressor", n_features=22)
m.compute(y_true, y_pred, y_baseline=lag1_values)  # y_baseline enables lift reporting
m.summary()       # statsmodels-style text table
m.to_dict()
m.to_dataframe()
# Attributes: mae, rmse, wape, mape, bias, r2, adj_r2
# With baseline: baseline_mae, baseline_wape, mae_lift_pct, wape_lift_pct
```

### `ClassificationMetrics` — src/utils/classification_metrics.py
```python
m = ClassificationMetrics(model_name="LGBMRegressor", beta=2.0)
m.compute(y_true, y_pred_proba, threshold=0.5,
          X_train=..., X_test=...)   # X_train/test optional, triggers adversarial AUC
m.summary()
m.find_optimal_threshold(y_true, y_pred_proba, metric="fbeta")
# Attributes: precision, recall, f1, fbeta, roc_auc, pr_auc, ks_stat, adversarial_auc
# tp, fp, fn, tn
```
Recall is prioritised over precision (missing a real peak costs more than a false alarm).

### `ConsumptionPlotter` — src/utils/plotter.py
All methods return `go.Figure`. Call `.show()` or pass `save_path` for HTML export.
```python
p = ConsumptionPlotter(model_name="LGBMRegressor")
p.plot_forecast(df, date_col, actual_col, pred_col, threshold_col, plant, sku, save_path)
p.plot_residuals(y_true, y_pred, dates)
p.plot_shap_importance(shap_values, feature_names, top_n=20)
p.plot_pr_curve(y_true, y_pred_proba, pr_auc)
p.plot_roc_curve(y_true, y_pred_proba, roc_auc)
p.plot_model_comparison(metrics_list, metric_cols=("mae","rmse","wape"))
p.plot_calibration(y_true, y_pred_proba, n_bins=10)
```

---

## Model candidate pool (from AGENT.md)

| Model | Class | Library | Status |
|---|---|---|---|
| LightGBM | `LGBMRegressor` | `lightgbm` | Production |
| TFT | `TFTRegressor` | `neuralforecast` | Candidate |
| XGBoost | — | `xgboost` | Candidate |
| CatBoost | — | `catboost` | Candidate |
| N-HiTS | — | `neuralforecast` | Candidate |
| N-BEATS | — | `neuralforecast` | Candidate |
| LightGBM Quantile | — | `lightgbm` | Planned |
| Conformal Prediction | — | `mapie` | Planned |

Rule: use `LGBMRegressor` for series with < 90 days of history. Switch to TFT or sequence models when >= 90 days available. Add probabilistic layer when confidence intervals are needed.

---

## Data schema (post build_features)

Key columns always present: `planta`, `sku`, `fecha`, `consumo_real`, `forecast_mensual`, `daily_forecast`, `target_is_peak`, all 22 feature columns.

`target_is_peak` is the binary label for Stage 2 evaluation. The regression target for Stage 1 is `consumo_real` (or `target_consumo_real` in some scripts — same column, different alias).

---

## Dependencies

Core: `polars`, `pandas`, `numpy`, `lightgbm`, `scikit-learn`, `plotly`, `shap`, `torch`
TFT: `neuralforecast>=1.7.0`
See `requirements.txt` for pinned versions.
