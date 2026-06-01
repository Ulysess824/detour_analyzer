# detour_analyzer Agent Guidelines

This document details the guidelines, rules, and process standards for modifying or developing components in this repository.

## General Coding Standards

- **Language**: All code and inline comments must be written in English.
- **Python Version**: Compatibility with Python 3.10+.
- **Formatting**:
  - No emojis in code print statements or markdown files to maintain a professional profile.
  - Use underscores as a thousands separator for numbers in scripts and notebooks (e.g., `1_000` instead of `1000`).
- **Modifications**:
  - Every modification to a class or method must include a usage example script in the chat.
  - Create a simple test script with synthetic data for any modified class/method.
  - After debugging, remove any temporary debugging scripts.

## Statistical and Validation Rigor

- **Official Statistics Standards**: Align with GSBPM (Generic Statistical Business Process Model) and GSIM (Generic Statistical Information Model) frameworks.
- **Validation**: Verify economic rationality (Engel curves, Walras' Law/budget constraints) and statistical performance via the metrics defined in the Model Evaluation section.

## Model Evaluation

All models predicting daily consumption must be evaluated under a **walk-forward (time-series split) cross-validation** scheme. Never use random splits — they leak future data into training.

### Regression Metrics (Stage 1)

| Metric | Formula | Threshold |
|---|---|---|
| MAE | mean(|y - y_hat|) | domain-specific baseline |
| RMSE | sqrt(mean((y - y_hat)^2)) | <= 1.2 * MAE (low outlier sensitivity) |
| WAPE | sum(|y - y_hat|) / sum(y) | < 0.20 target |
| MAPE | mean(|y - y_hat| / y) | only when y > 0; avoid zero-denominator |
| Bias | mean(y_hat - y) | should be near 0 (no systematic over/under) |

### Classification Metrics (Stage 2 — Peak Detection)

| Metric | Rationale |
|---|---|
| Precision / Recall | Recall is prioritized: missing a real peak is costlier than a false alarm |
| F1-Score | Harmonic mean; use F-beta (beta > 1) to weight recall more heavily |
| KS Statistic | Separation between peak and non-peak score distributions |
| Adversarial AUC | Detects train/test distribution shift; target AUC near 0.5 |
| PR-AUC | Preferred over ROC-AUC under class imbalance |

### Interpretability

- Compute **SHAP values** for every tabular model. Report mean absolute SHAP per feature.
- For sequence models, compute **attention weights** or **integrated gradients** to verify that the model attends to economically meaningful lags.

### Benchmark Baseline

Every model must beat a naive baseline: the previous day's actual consumption (lag-1 persistence). Report the percentage improvement over this baseline for MAE and WAPE.

## Project Context

This project focuses on a two-stage prediction system for daily consumption by plant and SKU.

### Data Structure and Definition
- **Sources**: Daily consumption database per plant and SKU.
- **Forecast**: Monthly forecast assigned to each SKU per plant. The expected daily forecast is computed by dividing the monthly forecast by 20 working days.
- **Peak Definition (Pico)**: A peak occurs when the daily actual consumption exceeds the expected daily forecast.

### Two-Stage Prediction Pipeline
- **Stage 1 (Regression)**: Predicts the exact magnitude of daily consumption. Predicting a continuous variable preserves the signal strength and avoids the noise of artificial binary thresholds.
- **Stage 2 (Derived Classification)**: The occurrence of a peak is classified dynamically by comparing the regression output with the dynamic threshold (e.g., `2.0 * daily_forecast`).

### Models and Architectures

The project maintains a ranked candidate pool. All candidates compete under the same walk-forward evaluation protocol. The current production model is LightGBM; others are in experimental or planned status.

#### Tabular / Gradient Boosting

| Model | Library | Status | Notes |
|---|---|---|---|
| **LightGBM** | `lightgbm` | Production | Fast, handles sparse SKUs well; SHAP-native |
| **XGBoost** | `xgboost` | Candidate | Strong regularization; use `tree_method="hist"` for speed |
| **CatBoost** | `catboost` | Candidate | Native categorical encoding; competitive on low-data SKUs |
| **TabNet** | `pytorch-tabnet` | Experimental | Attention-based tabular net; built-in feature selection |

#### Sequence / Deep Learning

| Model | Library | Status | Notes |
|---|---|---|---|
| **LSTM (hybrid)** | `torch` | Production | Models daily consumption sequences per SKU |
| **Temporal Fusion Transformer (TFT)** | `pytorch-forecasting` | Candidate | Handles multi-horizon, covariates, and interpretable attention |
| **N-HiTS** | `neuralforecast` | Candidate | Hierarchical interpolation; state-of-the-art on long horizons |
| **N-BEATS** | `neuralforecast` | Candidate | Generic basis expansion; strong univariate baseline |
| **TimesNet** | `pytorch` | Experimental | 2D temporal variation modeling via FFT-based period detection |

#### Probabilistic / Uncertainty-Aware

| Model | Library | Status | Notes |
|---|---|---|---|
| **LightGBM Quantile** | `lightgbm` | Planned | Pinball loss at q=0.90/0.95 for high-risk threshold alerts |
| **Conformal Prediction wrapper** | `mapie` | Planned | Distribution-free prediction intervals over any base model |
| **DeepAR** | `gluonts` | Experimental | Autoregressive RNN with parametric output distribution |

#### Model Selection Rules

- **Default choice**: LightGBM for new SKU/plant combinations due to sample efficiency.
- **Switch to sequence models** when a SKU has >= 90 consecutive trading days of history.
- **Add probabilistic layer** when the downstream decision (e.g., safety stock) requires a confidence interval, not just a point estimate.
- All models must expose a `predict(X) -> np.ndarray` interface. Wrap non-sklearn models with a thin adapter class.

### Feature Engineering (22 features in Polars)

- Lags: Actual consumption lags of 1, 2, 3, 5, and 10 days.
- Rolling Windows: Rolling mean, standard deviation, and maximum.
- Calendar Features: Day of the month and proximity to the end of the month.
- Deviation Features: Cumulative monthly forecast deviation.
- Historical SKU Profile: Historical peak rate and coefficient of variation.

### Future Improvements
- **Quantile Regression**: Train the model on the 90th or 95th percentile to alert on high-risk consumption thresholds directly rather than predicting the mean demand.
- **Kalman Filter**: Incorporate adaptive dynamic estimation to replace the static daily forecast, improving peak detection and reducing false positives in irregular consumption patterns.
- **Foundation model fine-tuning**: Evaluate TimesFM (Google) or Moirai (Salesforce) zero-shot forecasts as a cheap baseline before any training.
